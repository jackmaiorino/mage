from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from mtg_kernel_rl.artifacts import read_json_file
from mtg_kernel_rl.checkpoint import load_checkpoint_file
from mtg_kernel_rl.model import KernelPolicyValueNet
from mtg_kernel_rl.trainer import _compute_loss_tensors, train

from fixtures import fake_launcher


def _state(path: Path, update: int) -> dict:
    return load_checkpoint_file(path / "checkpoints" / f"update-{update:08d}.pt")


def _assert_tensor_map_equal(test: unittest.TestCase, a: dict, b: dict, prefix: str) -> None:
    test.assertEqual(set(a), set(b))
    for key in a:
        if isinstance(a[key], torch.Tensor):
            test.assertTrue(torch.equal(a[key], b[key]), f"{prefix}.{key}")
        elif isinstance(a[key], dict):
            _assert_tensor_map_equal(test, a[key], b[key], f"{prefix}.{key}")
        else:
            test.assertEqual(a[key], b[key], f"{prefix}.{key}")


class TrainerTest(unittest.TestCase):
    def test_fresh_update_zero_is_published_before_training_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            out = tmp / "run"
            result = train(
                env_bin=launcher,
                out_dir=out,
                base_seed=71501,
                until_update=0,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            self.assertEqual(result["completed_update"], 0)
            self.assertEqual(read_json_file(out / "latest.json")["update"], 0)
            self.assertEqual((out / "episodes.jsonl").read_text(encoding="utf-8"), "")
            self.assertTrue((out / "checkpoints" / "update-00000000.pt").is_file())

    def test_terminal_loss_matches_hand_computation_and_detaches_advantage(self) -> None:
        logp_a = torch.tensor(math.log(0.25), dtype=torch.float32, requires_grad=True)
        value_a = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
        logp_b = torch.tensor(math.log(0.5), dtype=torch.float32, requires_grad=True)
        value_b = torch.tensor(-0.1, dtype=torch.float32, requires_grad=True)
        logp_c = torch.tensor(math.log(0.25), dtype=torch.float32, requires_grad=True)
        value_c = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        terms = [(logp_a, value_a, 1), (logp_b, value_b, -1), (logp_c, value_c, 0)]
        policy_sum, value_sum, loss = _compute_loss_tensors(terms, 0.5)
        expected_policy = -math.log(0.25) * (1 - 0.2) - math.log(0.5) * (-1 + 0.1) - math.log(0.25) * 0
        expected_value = (0.2 - 1) ** 2 + (-0.1 + 1) ** 2 + 0.0**2
        expected_loss = (expected_policy + 0.5 * expected_value) / 3
        self.assertAlmostEqual(float(policy_sum.item()), expected_policy, places=6)
        self.assertAlmostEqual(float(value_sum.item()), expected_value, places=6)
        self.assertAlmostEqual(float(loss.item()), expected_loss, places=6)
        policy_only = _compute_loss_tensors(terms, 0.0)[2]
        policy_only.backward()
        self.assertAlmostEqual(float(value_a.grad.item()), 0.0, places=7)
        self.assertAlmostEqual(float(value_b.grad.item()), 0.0, places=7)

    def test_opponent_has_no_model_forwards_and_both_learner_seats_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            out = tmp / "run"
            calls = 0
            original = KernelPolicyValueNet.forward

            def wrapped(model: KernelPolicyValueNet, encoded):  # type: ignore[no-untyped-def]
                nonlocal calls
                calls += 1
                return original(model, encoded)

            KernelPolicyValueNet.forward = wrapped  # type: ignore[method-assign]
            try:
                train(
                    env_bin=launcher,
                    out_dir=out,
                    base_seed=71501,
                    until_update=1,
                    batch_episodes=2,
                    learning_rate=0.001,
                    value_coef=0.5,
                    max_decisions=8,
                )
            finally:
                KernelPolicyValueNet.forward = original  # type: ignore[method-assign]
            self.assertEqual(calls, 2)
            record = read_json_file(out / "updates" / "update-00000001.json")
            self.assertTrue(record["optimizer_step"])
            self.assertEqual([row["learner_seat"] for row in record["episode_summaries"]], ["p0", "p1"])
            self.assertEqual([row["learner_decision_count"] for row in record["episode_summaries"]], [1, 1])

    def test_zero_decision_batch_commits_without_model_or_optimizer_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_zero_learner")
            out = tmp / "run"
            result = train(
                env_bin=launcher,
                out_dir=out,
                base_seed=71501,
                until_update=1,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            self.assertEqual(result["completed_update"], 1)
            update0 = _state(out, 0)
            update1 = _state(out, 1)
            _assert_tensor_map_equal(self, update0["model_state"], update1["model_state"], "model")
            self.assertEqual(update0["optimizer_state"], update1["optimizer_state"])
            record = read_json_file(out / "updates" / "update-00000001.json")
            self.assertFalse(record["optimizer_step"])
            self.assertEqual(record["learner_decision_count"], 0)
            self.assertEqual(record["loss"], {"policy_sum_hex": None, "value_sum_hex": None, "loss_hex": None})

    def test_later_episode_failure_leaves_latest_at_prior_head_and_no_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_late_fault")
            out = tmp / "run"
            with self.assertRaises(Exception):
                train(
                    env_bin=launcher,
                    out_dir=out,
                    base_seed=71501,
                    until_update=1,
                    batch_episodes=2,
                    learning_rate=0.001,
                    value_coef=0.5,
                    max_decisions=8,
                )
            self.assertEqual(read_json_file(out / "latest.json")["update"], 0)
            self.assertEqual((out / "episodes.jsonl").read_text(encoding="utf-8"), "")
            self.assertFalse((out / "updates" / "update-00000001.json").exists())

    def test_uninterrupted_and_split_resume_are_exact_and_can_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            fresh = tmp / "fresh"
            split = tmp / "split"
            result_fresh = train(
                env_bin=launcher,
                out_dir=fresh,
                base_seed=71501,
                until_update=4,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            train(
                env_bin=launcher,
                out_dir=split,
                base_seed=71501,
                until_update=2,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            result_split = train(env_bin=launcher, out_dir=split, resume=split / "latest.json", until_update=4)
            self.assertEqual(result_fresh, result_split)
            _assert_tensor_map_equal(self, _state(fresh, 4)["model_state"], _state(split, 4)["model_state"], "model")
            _assert_tensor_map_equal(self, _state(fresh, 4)["optimizer_state"], _state(split, 4)["optimizer_state"], "optimizer")
            self.assertEqual((fresh / "updates.jsonl").read_text(encoding="utf-8"), (split / "updates.jsonl").read_text(encoding="utf-8"))
            result_fresh_5 = train(env_bin=launcher, out_dir=fresh, resume=fresh / "latest.json", until_update=5)
            result_split_5 = train(env_bin=launcher, out_dir=split, resume=split / "latest.json", until_update=5)
            self.assertEqual(result_fresh_5, result_split_5)

    def test_resume_noop_repairs_caches_and_rejects_conflicting_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            out = tmp / "run"
            train(
                env_bin=launcher,
                out_dir=out,
                base_seed=71501,
                until_update=1,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            (out / "episodes.jsonl").unlink()
            result = train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1)
            self.assertEqual(result["completed_update"], 1)
            self.assertTrue((out / "episodes.jsonl").is_file())
            with self.assertRaises(ValueError):
                train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1, learning_rate=0.002)


if __name__ == "__main__":
    unittest.main()
