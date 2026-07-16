from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch

from mtg_kernel_rl.client import Decision, KernelRlClient
from mtg_kernel_rl.determinism import derive_env_seed, derive_uniform_index
from mtg_kernel_rl.features import EDGE_ROLES, assert_action_classified, assert_observation_classified, encode_decision
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
            self.assertEqual(decision.provenance["protocol_version"], 3)
            self.assertEqual(decision.provenance["schema_version"], 3)
            self.assertEqual(decision.observation["schema_version"], 3)
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

    def test_release_kernel_env_complete_episode_one_persistent_model(self) -> None:
        env_bin = os.environ.get("MTG_KERNEL_RL_ENV_BIN")
        if not env_bin:
            self.skipTest("MTG_KERNEL_RL_ENV_BIN not set")
        self.assertTrue(Path(env_bin).is_file())
        model = None
        decisions = 0
        stage_tuples = set()
        edge_roles = set()
        with KernelRlClient(env_bin, timeout_s=10.0) as client:
            current = client.reset(episode_id=7, env_seed=derive_env_seed(71501, 7), max_decisions=5000)
            while isinstance(current, Decision):
                assert_observation_classified(current.observation)
                for action in current.legal_actions:
                    assert_action_classified(action)
                encoded = encode_decision(current.observation, current.legal_actions)
                if model is None:
                    model = KernelPolicyValueNet.from_encoded(encoded)
                logits, value = model(encoded)
                self.assertEqual(tuple(logits.shape), (len(current.legal_actions),))
                self.assertEqual(tuple(value.shape), ())
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(value).all())
                stage_tuples.add((current.observation["projection"]["engine_context"]["current_stage"], current.observation["projection"]["surface_context"]["current_stage"]))
                for row in encoded.edge_features:
                    role_index = int(torch.argmax(row[: len(EDGE_ROLES)]).item())
                    if float(row[role_index].item()) == 1.0:
                        edge_roles.add(EDGE_ROLES[role_index])
                selected = derive_uniform_index(71501, 7, current.step, current.acting_player, len(current.legal_actions))
                action = current.legal_actions[selected]
                current = client.step(action["selected_index"], action["stable_id"])
                decisions += 1
        self.assertIsNotNone(model)
        self.assertGreater(decisions, 0)
        self.assertEqual(current.terminal_classification, "natural")
        self.assertEqual(current.terminal_code, "natural_game_over")
        self.assertNotIn(current.terminal_outcome, {"halted", "truncated"})
        self.assertEqual(current.decision_count, decisions)
        self.assertIn(("priority", "priority"), stage_tuples)
        self.assertGreaterEqual(len(edge_roles), 1)


if __name__ == "__main__":
    unittest.main()
