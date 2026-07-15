from __future__ import annotations

import math
import os
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

from mtg_kernel_rl.artifacts import read_json_file, set_fault_injector, write_json_atomic
from mtg_kernel_rl.checkpoint import load_checkpoint_file, save_checkpoint_file
from mtg_kernel_rl.output_lock import OutputLock, OutputLockError
from mtg_kernel_rl.path_safety import filesystem_file_identity
from mtg_kernel_rl.model import KernelPolicyValueNet
from mtg_kernel_rl.trainer import _compute_loss_tensors, train
import mtg_kernel_rl.trainer as trainer_mod

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


def _assert_generation_equal(test: unittest.TestCase, left: Path, right: Path, update: int) -> None:
    _assert_tensor_map_equal(test, _state(left, update), _state(right, update), f"payload{update}")
    test.assertEqual(
        read_json_file(left / "updates" / f"update-{update:08d}.json"),
        read_json_file(right / "updates" / f"update-{update:08d}.json"),
    )
    test.assertEqual(
        read_json_file(left / "checkpoints" / f"update-{update:08d}.json"),
        read_json_file(right / "checkpoints" / f"update-{update:08d}.json"),
    )


def _assert_generation_logical_equal(test: unittest.TestCase, left: Path, right: Path, update: int) -> None:
    _assert_tensor_map_equal(test, _state(left, update), _state(right, update), f"payload{update}")
    left_update = read_json_file(left / "updates" / f"update-{update:08d}.json")
    right_update = read_json_file(right / "updates" / f"update-{update:08d}.json")
    left_update = {key: value for key, value in left_update.items() if key != "parent_head"}
    right_update = {key: value for key, value in right_update.items() if key != "parent_head"}
    test.assertEqual(left_update, right_update)
    left_sidecar = read_json_file(left / "checkpoints" / f"update-{update:08d}.json")
    right_sidecar = read_json_file(right / "checkpoints" / f"update-{update:08d}.json")
    for key in ("schema", "update", "run_digest", "logical_state_sha256"):
        test.assertEqual(left_sidecar[key], right_sidecar[key], key)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(["kernel/python", "kernel/python/tests"])
    return env


def _run_hard_kill_child(tmp: Path, out: Path, launcher: Path, boundary: str, target_fragment: str = "-") -> tuple[subprocess.CompletedProcess, Path]:
    script = tmp / f"hard_kill_{boundary}_{target_fragment.replace('-', 'none').replace('.', '_')}.py"
    marker = script.with_suffix(".marker.json")
    script.write_text(
        """
import json
import os
import sys
from pathlib import Path

from mtg_kernel_rl.artifacts import set_fault_injector
from mtg_kernel_rl.trainer import train

boundary = sys.argv[1]
target_fragment = sys.argv[2]
out = Path(sys.argv[3])
launcher = Path(sys.argv[4])
marker = Path(sys.argv[5])

def injector(name, path):
    if name != boundary:
        return
    if target_fragment != "-":
        if path is None or target_fragment not in Path(path).name:
            return
    marker.write_text(json.dumps({"boundary": name, "path": None if path is None else Path(path).name}, sort_keys=True), encoding="utf-8")
    os._exit(73)

set_fault_injector(injector)
train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1)
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script), boundary, target_fragment, str(out), str(launcher), str(marker)],
        cwd=Path.cwd(),
        env=_subprocess_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed, marker


def _run_hard_kill_fresh_child(tmp: Path, out: Path, launcher: Path, boundary: str, target_fragment: str = "-", until_update: int = 0) -> tuple[subprocess.CompletedProcess, Path]:
    script = tmp / f"hard_kill_fresh_{boundary}_{target_fragment.replace('-', 'none').replace('.', '_')}.py"
    marker = script.with_suffix(".marker.json")
    script.write_text(
        """
import json
import os
import sys
from pathlib import Path

from mtg_kernel_rl.artifacts import set_fault_injector
from mtg_kernel_rl.trainer import train

boundary = sys.argv[1]
target_fragment = sys.argv[2]
out = Path(sys.argv[3])
launcher = Path(sys.argv[4])
marker = Path(sys.argv[5])
until_update = int(sys.argv[6])

def injector(name, path):
    if name != boundary:
        return
    if target_fragment != "-":
        if path is None or target_fragment not in Path(path).name:
            return
    marker.write_text(json.dumps({"boundary": name, "path": None if path is None else Path(path).name}, sort_keys=True), encoding="utf-8")
    os._exit(73)

set_fault_injector(injector)
train(
    env_bin=launcher,
    out_dir=out,
    base_seed=71501,
    until_update=until_update,
    batch_episodes=2,
    learning_rate=0.001,
    value_coef=0.5,
    max_decisions=8,
)
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script), boundary, target_fragment, str(out), str(launcher), str(marker), str(until_update)],
        cwd=Path.cwd(),
        env=_subprocess_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed, marker


def _assert_hard_kill_fired(test: unittest.TestCase, child: subprocess.CompletedProcess, marker: Path, boundary: str) -> None:
    test.assertEqual(child.returncode, 73)
    test.assertTrue(marker.is_file(), f"missing marker for {boundary}")
    data = json.loads(marker.read_text(encoding="utf-8"))
    test.assertEqual(data["boundary"], boundary)


class TrainerTest(unittest.TestCase):
    def test_fresh_update_zero_is_published_before_training_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = tmp / "run"
            launcher = fake_launcher(tmp, "train_pair_assert_latest0", {"FAKE_EXPECT_LATEST_JSON": str(out / "latest.json")})
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
            self.assertEqual(read_json_file(out / "updates" / "update-00000000.json")["update"], 0)
            self.assertTrue((out / "checkpoints" / "update-00000000.pt").is_file())
            run = read_json_file(out / "run.json")
            self.assertEqual(run["schema"], "kernel_rl_train_run/v5")
            self.assertEqual(run["artifact_boundary"]["schema"], "kernel_rl_artifact_boundary/v3")

    def test_fresh_reset_failure_and_pre_manifest_debris_are_recoverable_or_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = tmp / "run"
            failing = fake_launcher(tmp, "eof_nonzero")
            launcher = fake_launcher(tmp, "train_pair")
            control = tmp / "control"
            control_result = train(
                env_bin=launcher,
                out_dir=control,
                base_seed=71501,
                until_update=2,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            with self.assertRaises(Exception):
                train(
                    env_bin=failing,
                    out_dir=out,
                    base_seed=71501,
                    until_update=0,
                    batch_episodes=2,
                    learning_rate=0.001,
                    value_coef=0.5,
                    max_decisions=8,
                )
            recovered = train(
                env_bin=launcher,
                out_dir=out,
                base_seed=71501,
                until_update=2,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            self.assertEqual(recovered, control_result)
            _assert_generation_equal(self, out, control, 2)

            clean_debris = tmp / "clean_debris"
            clean_debris.mkdir()
            (clean_debris / "updates").mkdir()
            (clean_debris / "checkpoints").mkdir()
            result = train(
                env_bin=launcher,
                out_dir=clean_debris,
                base_seed=71501,
                until_update=0,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            self.assertEqual(result["completed_update"], 0)

            unknown = tmp / "unknown_debris"
            unknown.mkdir()
            known = unknown / "updates"
            known.mkdir()
            before = known.stat()
            (unknown / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                train(
                    env_bin=launcher,
                    out_dir=unknown,
                    base_seed=71501,
                    until_update=0,
                    batch_episodes=2,
                    learning_rate=0.001,
                    value_coef=0.5,
                    max_decisions=8,
                )
            self.assertTrue((unknown / "unexpected.txt").exists())
            after = known.stat()
            self.assertEqual((before.st_dev, before.st_ino, before.st_mtime_ns), (after.st_dev, after.st_ino, after.st_mtime_ns))
            self.assertTrue(known.is_dir())

    def test_fresh_run_json_hard_death_reuses_same_output_path(self) -> None:
        cases = [
            ("json_save", "run.json"),
            ("json_flush", "run.json"),
            ("json_fsync", "run.json"),
            ("json_replace_before", "run.json"),
            ("json_replace_after", "run.json"),
        ]
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            control = tmp / "control"
            train(
                env_bin=launcher,
                out_dir=control,
                base_seed=71501,
                until_update=2,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            for index, (boundary, target) in enumerate(cases):
                with self.subTest(boundary=boundary, target=target):
                    out = tmp / f"runjson_{index}"
                    child, marker = _run_hard_kill_fresh_child(tmp, out, launcher, boundary, target, until_update=0)
                    _assert_hard_kill_fired(self, child, marker, boundary)
                    if (out / "latest.json").exists():
                        recovered = train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=2)
                    else:
                        recovered = train(
                            env_bin=launcher,
                            out_dir=out,
                            base_seed=71501,
                            until_update=2,
                            batch_episodes=2,
                            learning_rate=0.001,
                            value_coef=0.5,
                            max_decisions=8,
                        )
                    self.assertEqual(recovered["completed_update"], 2)
                    _assert_generation_logical_equal(self, out, control, 0)
                    _assert_generation_logical_equal(self, out, control, 2)

    def test_fresh_update_zero_hard_death_matrix_recovers_and_continues(self) -> None:
        cases = [
            ("checkpoint_save", "-", False),
            ("checkpoint_flush", "-", False),
            ("checkpoint_fsync", "-", False),
            ("json_save", "update.json", False),
            ("json_flush", "update.json", False),
            ("json_fsync", "update.json", False),
            ("json_replace_after", "update.json", False),
            ("json_save", "sidecar.json", False),
            ("json_flush", "sidecar.json", False),
            ("json_fsync", "sidecar.json", False),
            ("json_replace_after", "sidecar.json", False),
            ("generation_validate", "-", False),
            ("final_replace_checkpoint_before", "-", False),
            ("final_replace_checkpoint_after", "-", False),
            ("final_replace_update_before", "-", False),
            ("final_replace_update_after", "-", False),
            ("final_replace_sidecar_before", "-", False),
            ("final_replace_sidecar_after", "-", False),
            ("json_save", "latest.json", False),
            ("json_flush", "latest.json", False),
            ("json_fsync", "latest.json", False),
            ("json_replace_after", "latest.json", True),
            ("latest_replace_before", "-", False),
            ("latest_replace_after", "-", True),
            ("post_latest_cleanup_before", "-", True),
        ]
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            control = tmp / "control"
            train(
                env_bin=launcher,
                out_dir=control,
                base_seed=71501,
                until_update=2,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            for index, (boundary, target, latest_exists) in enumerate(cases):
                with self.subTest(boundary=boundary, target=target):
                    out = tmp / f"update0_{index}"
                    child, marker = _run_hard_kill_fresh_child(tmp, out, launcher, boundary, target, until_update=0)
                    _assert_hard_kill_fired(self, child, marker, boundary)
                    self.assertEqual((out / "latest.json").exists(), latest_exists)
                    if latest_exists:
                        recovered = train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=2)
                    else:
                        recovered = train(
                            env_bin=launcher,
                            out_dir=out,
                            base_seed=71501,
                            until_update=2,
                            batch_episodes=2,
                            learning_rate=0.001,
                            value_coef=0.5,
                            max_decisions=8,
                        )
                    self.assertEqual(recovered["completed_update"], 2)
                    _assert_generation_logical_equal(self, out, control, 0)
                    _assert_generation_logical_equal(self, out, control, 2)

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
            hook_grads: list[tuple[int, torch.Tensor]] = []
            original = KernelPolicyValueNet.forward

            def wrapped(model: KernelPolicyValueNet, encoded):  # type: ignore[no-untyped-def]
                nonlocal calls
                calls += 1
                logits, value = original(model, encoded)
                hook_index = calls
                logits.register_hook(lambda grad, index=hook_index: hook_grads.append((index, grad.detach().clone())))
                return logits, value

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
            self.assertEqual(sorted(index for index, _grad in hook_grads), [1, 2])
            for _index, grad in hook_grads:
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(torch.sum(torch.abs(grad)).item()), 0.0)
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
            with self.assertRaises(TypeError):
                train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1, learning_rate=1)

    def test_transaction_faults_before_latest_replay_from_old_head(self) -> None:
        boundaries = [
            "checkpoint_save",
            "checkpoint_flush",
            "checkpoint_fsync",
            "json_save",
            "json_flush",
            "json_fsync",
            "generation_validate",
            "final_replace_checkpoint_after",
            "final_replace_update_after",
            "final_replace_sidecar_after",
            "latest_replace_before",
        ]
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            clean = tmp / "clean"
            clean_result = train(
                env_bin=launcher,
                out_dir=clean,
                base_seed=71501,
                until_update=1,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            for boundary in boundaries:
                with self.subTest(boundary=boundary):
                    out = tmp / f"fault_{boundary}"
                    train(
                        env_bin=launcher,
                        out_dir=out,
                        base_seed=71501,
                        until_update=0,
                        batch_episodes=2,
                        learning_rate=0.001,
                        value_coef=0.5,
                        max_decisions=8,
                    )
                    fired = {"value": False}

                    def injector(name: str, _path: Path | None) -> None:
                        if name == boundary and not fired["value"]:
                            fired["value"] = True
                            raise RuntimeError(f"fault at {boundary}")

                    previous = set_fault_injector(injector)
                    try:
                        with self.assertRaises(RuntimeError):
                            train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1)
                    finally:
                        set_fault_injector(previous)
                    self.assertTrue(fired["value"])
                    self.assertEqual(read_json_file(out / "latest.json")["update"], 0)
                    recovered = train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1)
                    self.assertEqual(recovered, clean_result)
                    self.assertEqual(read_json_file(out / "latest.json")["update"], 1)

    def test_latest_after_fault_leaves_new_complete_head_and_rebuilds_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            out = tmp / "run"
            train(
                env_bin=launcher,
                out_dir=out,
                base_seed=71501,
                until_update=0,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            fired = {"value": False}

            def injector(name: str, _path: Path | None) -> None:
                if name == "latest_replace_after" and not fired["value"]:
                    fired["value"] = True
                    raise RuntimeError("post-latest fault")

            previous = set_fault_injector(injector)
            try:
                with self.assertRaises(RuntimeError):
                    train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1)
            finally:
                set_fault_injector(previous)
            self.assertTrue(fired["value"])
            self.assertEqual(read_json_file(out / "latest.json")["update"], 1)
            if (out / "episodes.jsonl").exists():
                (out / "episodes.jsonl").unlink()
            result = train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1)
            self.assertEqual(result["completed_update"], 1)
            self.assertTrue((out / "episodes.jsonl").is_file())

    def test_latest_after_os_exit_leaves_debris_and_recovers_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            control = tmp / "control"
            out = tmp / "run"
            train(
                env_bin=launcher,
                out_dir=control,
                base_seed=71501,
                until_update=2,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            train(
                env_bin=launcher,
                out_dir=out,
                base_seed=71501,
                until_update=0,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            child, marker = _run_hard_kill_child(tmp, out, launcher, "latest_replace_after")
            _assert_hard_kill_fired(self, child, marker, "latest_replace_after")
            self.assertEqual(read_json_file(out / "latest.json")["update"], 1)
            transactions = out / ".transactions"
            self.assertTrue(transactions.is_dir())
            self.assertTrue(any(child_dir.name.startswith("update-00000001-") for child_dir in transactions.iterdir()))
            recovered = train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1)
            self.assertEqual(recovered["completed_update"], 1)
            self.assertFalse(transactions.exists())
            _assert_generation_equal(self, out, control, 1)
            self.assertEqual((out / "updates.jsonl").read_text(encoding="utf-8"), "\n".join(
                (control / "updates.jsonl").read_text(encoding="utf-8").splitlines()[:2]
            ) + "\n")
            continued = train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=2)
            self.assertEqual(continued["completed_update"], 2)
            _assert_generation_equal(self, out, control, 2)
            self.assertEqual(continued, train(env_bin=launcher, out_dir=control, resume=control / "latest.json", until_update=2))

    def test_os_exit_crash_boundaries_recover_to_old_or_new_complete_head(self) -> None:
        cases = [
            ("checkpoint_save", "-", 0),
            ("checkpoint_flush", "-", 0),
            ("checkpoint_fsync", "-", 0),
            ("json_save", "update.json", 0),
            ("json_flush", "update.json", 0),
            ("json_fsync", "update.json", 0),
            ("json_replace_before", "update.json", 0),
            ("json_save", "sidecar.json", 0),
            ("json_flush", "sidecar.json", 0),
            ("json_fsync", "sidecar.json", 0),
            ("json_replace_before", "sidecar.json", 0),
            ("generation_validate", "-", 0),
            ("final_replace_checkpoint_before", "-", 0),
            ("final_replace_checkpoint_after", "-", 0),
            ("final_replace_update_before", "-", 0),
            ("final_replace_update_after", "-", 0),
            ("final_replace_sidecar_before", "-", 0),
            ("final_replace_sidecar_after", "-", 0),
            ("json_save", "latest.json", 0),
            ("json_flush", "latest.json", 0),
            ("json_fsync", "latest.json", 0),
            ("json_replace_before", "latest.json", 0),
            ("latest_replace_before", "-", 0),
            ("latest_replace_after", "-", 1),
            ("json_save", "episodes.jsonl", 0),
            ("json_flush", "episodes.jsonl", 0),
            ("json_fsync", "episodes.jsonl", 0),
            ("json_replace_before", "episodes.jsonl", 0),
            ("json_save", "updates.jsonl", 0),
            ("json_flush", "updates.jsonl", 0),
            ("json_fsync", "updates.jsonl", 0),
            ("json_replace_before", "updates.jsonl", 0),
            ("json_save", "summary.json", 0),
            ("json_flush", "summary.json", 0),
            ("json_fsync", "summary.json", 0),
            ("json_replace_before", "summary.json", 0),
            ("post_latest_cleanup_before", "-", 1),
        ]
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            control = tmp / "control"
            train(
                env_bin=launcher,
                out_dir=control,
                base_seed=71501,
                until_update=2,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            for index, (boundary, target, expected_head) in enumerate(cases):
                with self.subTest(boundary=boundary, target=target):
                    out = tmp / f"fault_{index}"
                    train(
                        env_bin=launcher,
                        out_dir=out,
                        base_seed=71501,
                        until_update=0,
                        batch_episodes=2,
                        learning_rate=0.001,
                        value_coef=0.5,
                        max_decisions=8,
                    )
                    child, marker = _run_hard_kill_child(tmp, out, launcher, boundary, target)
                    _assert_hard_kill_fired(self, child, marker, boundary)
                    self.assertEqual(read_json_file(out / "latest.json")["update"], expected_head)
                    recovered = train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=expected_head)
                    self.assertEqual(recovered["completed_update"], expected_head)
                    _assert_generation_equal(self, out, control, expected_head)
                    continued = train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=2)
                    self.assertEqual(continued["completed_update"], 2)
                    _assert_generation_equal(self, out, control, 2)

    def test_recognized_transaction_debris_is_removed_but_malformed_trees_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            source = tmp / "source"
            train(
                env_bin=launcher,
                out_dir=source,
                base_seed=71501,
                until_update=1,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            tx_root = source / ".transactions"
            reachable_tx = tx_root / "update-00000001-deadbeef"
            future_tx = tx_root / "update-00000002-feedface"
            reachable_tx.mkdir(parents=True)
            shutil.copy2(source / "checkpoints" / "update-00000001.pt", reachable_tx / "checkpoint.pt")
            shutil.copy2(source / "updates" / "update-00000001.json", reachable_tx / "update.json")
            shutil.copy2(source / "checkpoints" / "update-00000001.json", reachable_tx / "sidecar.json")
            future_tx.mkdir()
            (future_tx / ".update.json.1.2.tmp").write_text("{}", encoding="utf-8")
            (source / "episodes.jsonl").unlink()
            recovered = train(env_bin=launcher, out_dir=source, resume=source / "latest.json", until_update=1)
            self.assertEqual(recovered["completed_update"], 1)
            self.assertFalse(tx_root.exists())
            self.assertTrue((source / "episodes.jsonl").is_file())

            def malformed_case(name: str, builder) -> None:  # type: ignore[no-untyped-def]
                target = tmp / name
                shutil.copytree(source, target)
                builder(target)
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        train(env_bin=launcher, out_dir=target, resume=target / "latest.json", until_update=1)

            malformed_case("unknown_transaction_name", lambda p: (p / ".transactions" / "not-an-update").mkdir(parents=True))
            malformed_case(
                "nested_transaction_directory",
                lambda p: (p / ".transactions" / "update-00000001-deadbeef" / "nested").mkdir(parents=True),
            )
            malformed_case(
                "unknown_transaction_file",
                lambda p: (
                    (p / ".transactions" / "update-00000001-deadbeef").mkdir(parents=True),
                    (p / ".transactions" / "update-00000001-deadbeef" / "evil.bin").write_bytes(b"x"),
                ),
            )

            def escaping_link(p: Path) -> None:
                tx = p / ".transactions" / "update-00000001-deadbeef"
                tx.mkdir(parents=True)
                outside = tmp / "outside.pt"
                outside.write_bytes(b"outside")
                try:
                    os.symlink(outside, tx / "checkpoint.pt")
                except (OSError, NotImplementedError) as exc:
                    raise unittest.SkipTest(f"symlink unavailable: {exc}") from exc

            try:
                malformed_case("escaping_transaction_link", escaping_link)
            except unittest.SkipTest:
                pass

    def test_no_follow_rejects_artifact_links_and_preserves_external_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            source = tmp / "source"
            outside = tmp / "outside"
            outside.mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_bytes(b"external-sentinel")
            train(
                env_bin=launcher,
                out_dir=source,
                base_seed=71501,
                until_update=0,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )

            def run_case(name: str, mutator) -> None:  # type: ignore[no-untyped-def]
                target = tmp / name
                shutil.copytree(source, target)
                mutator(target)
                with self.subTest(name=name):
                    with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
                        train(env_bin=launcher, out_dir=target, resume=target / "latest.json", until_update=0)
                    self.assertEqual(sentinel.read_bytes(), b"external-sentinel")

            if hasattr(os, "symlink"):
                def latest_symlink(p: Path) -> None:
                    (p / "latest.json").unlink()
                    os.symlink(sentinel, p / "latest.json")

                try:
                    run_case("latest_symlink", latest_symlink)
                except (OSError, NotImplementedError):
                    pass

            def latest_hardlink(p: Path) -> None:
                outside_latest = outside / "latest-hardlink.json"
                outside_latest.write_bytes((p / "latest.json").read_bytes())
                (p / "latest.json").unlink()
                os.link(outside_latest, p / "latest.json")

            run_case("latest_hardlink", latest_hardlink)

            if os.name == "nt":
                def updates_junction(p: Path) -> None:
                    shutil.rmtree(p / "updates")
                    completed = subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(p / "updates"), str(outside)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    if completed.returncode != 0:
                        self.fail("Windows junction coverage could not create mklink /J")

                run_case("updates_junction", updates_junction)

    def test_malformed_rng_cannot_be_committed_or_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            out = tmp / "run"
            original = trainer_mod.build_checkpoint_payload

            def bad_payload(**kwargs):  # type: ignore[no-untyped-def]
                payload = original(**kwargs)
                if payload["completed_update"] == 1:
                    payload["torch_cpu_rng_state"] = torch.zeros_like(payload["torch_cpu_rng_state"])
                return payload

            trainer_mod.build_checkpoint_payload = bad_payload  # type: ignore[assignment]
            try:
                with self.assertRaises(ValueError):
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
                trainer_mod.build_checkpoint_payload = original  # type: ignore[assignment]
            self.assertEqual(read_json_file(out / "latest.json")["update"], 0)
            self.assertFalse((out / "updates" / "update-00000001.json").exists())

    def test_windows_replace_error_retries_twice_then_succeeds(self) -> None:
        import mtg_kernel_rl.artifacts as artifacts

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            out = tmp / "run"
            original = artifacts.os.replace
            calls = {"count": 0}

            def flaky_replace(src, dst):  # type: ignore[no-untyped-def]
                if Path(dst).name == "latest.json" and calls["count"] < 2:
                    calls["count"] += 1
                    exc = OSError("simulated sharing violation")
                    exc.winerror = 5  # type: ignore[attr-defined]
                    raise exc
                return original(src, dst)

            artifacts.os.replace = flaky_replace  # type: ignore[assignment]
            try:
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
            finally:
                artifacts.os.replace = original  # type: ignore[assignment]
            self.assertEqual(calls["count"], 2)
            self.assertEqual(result["completed_update"], 1)

    def test_corruption_matrix_rejects_direct_artifact_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            source = tmp / "source"
            train(
                env_bin=launcher,
                out_dir=source,
                base_seed=71501,
                until_update=1,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )

            def case(name: str, mutator) -> None:  # type: ignore[no-untyped-def]
                target = tmp / name
                shutil.copytree(source, target)
                mutator(target)
                with self.subTest(name=name):
                    with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
                        train(env_bin=launcher, out_dir=target, resume=target / "latest.json", until_update=1)

            case("run_extra_privacy_field", lambda p: write_json_atomic(p / "run.json", {**read_json_file(p / "run.json"), "observation": {}}))
            case("latest_extra_field", lambda p: write_json_atomic(p / "latest.json", {**read_json_file(p / "latest.json"), "extra": True}))
            case(
                "update_wrong_range",
                lambda p: write_json_atomic(
                    p / "updates" / "update-00000001.json",
                    {**read_json_file(p / "updates" / "update-00000001.json"), "episode_start": 2},
                ),
            )
            case(
                "update_privacy_field",
                lambda p: write_json_atomic(
                    p / "updates" / "update-00000001.json",
                    {**read_json_file(p / "updates" / "update-00000001.json"), "display_text": "Cast"},
                ),
            )
            case(
                "sidecar_wrong_parent",
                lambda p: write_json_atomic(
                    p / "checkpoints" / "update-00000001.json",
                    {**read_json_file(p / "checkpoints" / "update-00000001.json"), "parent_head": "0" * 64},
                ),
            )
            case("missing_reachable_generation", lambda p: (p / "updates" / "update-00000000.json").unlink())
            case("unknown_generation_name", lambda p: (p / "updates" / "update-1.json").write_text("{}", encoding="utf-8"))
            case("old_v4_run_schema_rejected", lambda p: write_json_atomic(p / "run.json", {**read_json_file(p / "run.json"), "schema": "kernel_rl_train_run/v4"}))
            case(
                "old_v2_artifact_boundary_rejected",
                lambda p: write_json_atomic(
                    p / "run.json",
                    {
                        **read_json_file(p / "run.json"),
                        "artifact_boundary": {
                            **read_json_file(p / "run.json")["artifact_boundary"],
                            "schema": "kernel_rl_artifact_boundary/v2",
                        },
                    },
                ),
            )
            case(
                "checkpoint_counter_drift",
                lambda p: (
                    (lambda payload: (
                        payload.__setitem__("next_episode", payload["next_episode"] + 2),
                        save_checkpoint_file(p / "checkpoints" / "update-00000001.pt", payload),
                    ))(load_checkpoint_file(p / "checkpoints" / "update-00000001.pt"))
                ),
            )

    def test_subprocess_continuous_split_and_future_exact_without_pt_byte_equality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            fresh = tmp / "fresh"
            split = tmp / "split"
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(["kernel/python", "kernel/python/tests"])

            def run_train(args: list[str]) -> None:
                subprocess.check_call(
                    [sys.executable, "-m", "mtg_kernel_rl", "train", *args],
                    cwd=Path.cwd(),
                    env=env,
                    stdout=subprocess.DEVNULL,
                )

            common = [
                "--env-bin",
                str(launcher),
                "--base-seed",
                "71501",
                "--batch-episodes",
                "2",
                "--learning-rate",
                "0.001",
                "--value-coef",
                "0.5",
                "--max-decisions",
                "8",
            ]
            run_train(["--out-dir", str(fresh), "--until-update", "4", *common])
            run_train(["--out-dir", str(split), "--until-update", "2", *common])
            run_train(["--env-bin", str(launcher), "--out-dir", str(split), "--resume", str(split / "latest.json"), "--until-update", "4"])
            run_train(["--env-bin", str(launcher), "--out-dir", str(fresh), "--resume", str(fresh / "latest.json"), "--until-update", "5"])
            run_train(["--env-bin", str(launcher), "--out-dir", str(split), "--resume", str(split / "latest.json"), "--until-update", "5"])
            for update in (4, 5):
                fresh_payload = _state(fresh, update)
                split_payload = _state(split, update)
                _assert_tensor_map_equal(self, fresh_payload["model_state"], split_payload["model_state"], f"model{update}")
                _assert_tensor_map_equal(self, fresh_payload["optimizer_state"], split_payload["optimizer_state"], f"optimizer{update}")
                self.assertEqual(fresh_payload["python_rng_state"], split_payload["python_rng_state"])
                self.assertTrue(torch.equal(fresh_payload["torch_cpu_rng_state"], split_payload["torch_cpu_rng_state"]))
                self.assertEqual(fresh_payload["completed_update"], split_payload["completed_update"])
                self.assertEqual(fresh_payload["optimizer_step_count"], split_payload["optimizer_step_count"])
                self.assertEqual(fresh_payload["next_episode"], split_payload["next_episode"])
                self.assertEqual(read_json_file(fresh / "updates" / f"update-{update:08d}.json"), read_json_file(split / "updates" / f"update-{update:08d}.json"))
            self.assertEqual((fresh / "episodes.jsonl").read_text(encoding="utf-8"), (split / "episodes.jsonl").read_text(encoding="utf-8"))

    def test_actor_local_sampling_ignores_unrelated_global_rng_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            clean = tmp / "clean"
            noisy = tmp / "noisy"
            clean_result = train(
                env_bin=launcher,
                out_dir=clean,
                base_seed=71501,
                until_update=3,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            random.seed(999)
            for _ in range(100):
                random.random()
            torch.manual_seed(999)
            _ = torch.rand(100)
            noisy_result = train(
                env_bin=launcher,
                out_dir=noisy,
                base_seed=71501,
                until_update=3,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            self.assertEqual(clean_result, noisy_result)
            self.assertEqual((clean / "updates.jsonl").read_text(encoding="utf-8"), (noisy / "updates.jsonl").read_text(encoding="utf-8"))

    def test_output_lock_alias_distinct_exception_and_noninheritance(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_name:
            tmp = Path(tmp_name)
            root = tmp / "run"
            alias = Path(os.path.relpath(root, Path.cwd()))
            with OutputLock(root) as lock:
                self.assertTrue(lock.path.is_file())
                with self.assertRaises(OutputLockError):
                    with OutputLock(alias):
                        pass
                with OutputLock(tmp / "other"):
                    pass

            script = tmp / "owner.py"
            marker = tmp / "owner.marker"
            script.write_text(
                """
import os
import sys
from pathlib import Path
from mtg_kernel_rl.output_lock import OutputLock

root = Path(sys.argv[1])
marker = Path(sys.argv[2])
with OutputLock(root) as lock:
    marker.write_text(str(lock.path), encoding="utf-8")
    os._exit(73)
""",
                encoding="utf-8",
            )
            child = subprocess.run([sys.executable, str(script), str(root), str(marker)], cwd=Path.cwd(), env=_subprocess_env(), check=False)
            self.assertEqual(child.returncode, 73)
            lock_path = Path(marker.read_text(encoding="utf-8"))
            self.assertTrue(lock_path.is_file())
            hard_death_identity = filesystem_file_identity(lock_path)
            with OutputLock(root):
                pass
            self.assertEqual(filesystem_file_identity(lock_path), hard_death_identity)

            noninherit = tmp / "noninherit.py"
            noninherit_marker = tmp / "noninherit.marker"
            noninherit.write_text(
                """
import subprocess
import sys
from pathlib import Path
from mtg_kernel_rl.output_lock import OutputLock

root = Path(sys.argv[1])
marker = Path(sys.argv[2])
with OutputLock(root):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], close_fds=False)
    marker.write_text(str(child.pid), encoding="utf-8")
""",
                encoding="utf-8",
            )
            subprocess.check_call([sys.executable, str(noninherit), str(root), str(noninherit_marker)], cwd=Path.cwd(), env=_subprocess_env())
            with OutputLock(root):
                pass

            long_parent = tmp / "LongPhysicalAliasParent"
            long_parent.mkdir()
            long_root = long_parent / "LongPhysicalAliasRun"
            long_root.mkdir()
            if os.name == "nt":
                junction_parent = tmp / "junction_parent"
                completed = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(junction_parent), str(long_parent)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if completed.returncode != 0:
                    self.fail("Windows junction coverage could not create mklink /J")
                physical_alias = junction_parent / long_root.name
            else:
                physical_alias = tmp / "symlink_parent" / long_root.name
                try:
                    os.symlink(long_parent, tmp / "symlink_parent", target_is_directory=True)
                except (OSError, NotImplementedError) as exc:
                    raise unittest.SkipTest(f"POSIX symlink ancestor unavailable: {exc}") from exc
            with OutputLock(long_root) as lock:
                with self.assertRaises(OutputLockError):
                    with OutputLock(physical_alias):
                        pass
                alias_lock = OutputLock(physical_alias)
                self.assertTrue(os.path.samefile(lock.path, alias_lock.path))
                self.assertEqual(filesystem_file_identity(lock.path), filesystem_file_identity(alias_lock.path))
                self.assertEqual(lock.path.name, ".mtg-kernel-train.lock")
                self.assertEqual(len(list(long_root.glob(".mtg-kernel-train.lock"))), 1)

            if os.name == "nt":
                cmd = f'for %I in ("{long_root}") do @echo %~sI'
                completed = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, check=False)
                short_text = completed.stdout.strip()
                capability = tmp / "short-name-capability.json"
                if completed.returncode == 0 and short_text and "~" in short_text and Path(short_text).exists():
                    capability.write_text(json.dumps({"short_name": "available", "path": short_text}, sort_keys=True), encoding="utf-8")
                    with OutputLock(long_root):
                        with self.assertRaises(OutputLockError):
                            with OutputLock(short_text):
                                pass
                else:
                    capability.write_text(json.dumps({"short_name": "unavailable", "path": short_text}, sort_keys=True), encoding="utf-8")
                    self.assertIn("short_name", read_json_file(capability, require_canonical=False))

    def test_same_root_concurrent_trainers_exclude_loser_without_second_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = tmp / "run"
            launcher = fake_launcher(tmp, "train_pair_slow")
            env = _subprocess_env()
            common = [
                sys.executable,
                "-m",
                "mtg_kernel_rl",
                "train",
                "--env-bin",
                str(launcher),
                "--out-dir",
                str(out),
                "--base-seed",
                "71501",
                "--batch-episodes",
                "2",
                "--learning-rate",
                "0.001",
                "--value-coef",
                "0.5",
                "--max-decisions",
                "8",
            ]
            owner_script = tmp / "lock_owner.py"
            owner_marker = tmp / "lock_owner.marker"
            release_marker = tmp / "lock_owner.release"
            owner_script.write_text(
                """
import sys
import time
from pathlib import Path
from mtg_kernel_rl.output_lock import OutputLock

root = Path(sys.argv[1])
marker = Path(sys.argv[2])
release = Path(sys.argv[3])
with OutputLock(root) as lock:
    marker.write_text(str(lock.path), encoding="utf-8")
    while not release.exists():
        time.sleep(0.05)
""",
                encoding="utf-8",
            )
            owner = subprocess.Popen([sys.executable, str(owner_script), str(out), str(owner_marker), str(release_marker)], cwd=Path.cwd(), env=env)
            try:
                deadline = time.time() + 10
                while not owner_marker.exists() and time.time() < deadline:
                    time.sleep(0.05)
                self.assertTrue(owner_marker.exists())
                lock_path = Path(owner_marker.read_text(encoding="utf-8"))
                loser_marker = tmp / "loser_after_train.marker"
                loser_script = tmp / "lock_loser.py"
                loser_script.write_text(
                    """
import sys
from pathlib import Path
from mtg_kernel_rl.output_lock import OutputLockError
from mtg_kernel_rl.trainer import train

launcher = Path(sys.argv[1])
out = Path(sys.argv[2])
marker = Path(sys.argv[3])
try:
    train(
        env_bin=launcher,
        out_dir=out,
        base_seed=71501,
        until_update=0,
        batch_episodes=2,
        learning_rate=0.001,
        value_coef=0.5,
        max_decisions=8,
    )
except OutputLockError:
    sys.exit(73)
marker.write_text("launched", encoding="utf-8")
sys.exit(0)
""",
                    encoding="utf-8",
                )
                loser = subprocess.run(
                    [sys.executable, str(loser_script), str(launcher), str(out), str(loser_marker)],
                    cwd=Path.cwd(),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                self.assertEqual(loser.returncode, 73)
                self.assertFalse(loser_marker.exists())
                self.assertFalse((out / "run.json").exists())
                self.assertEqual(lock_path.name, ".mtg-kernel-train.lock")
                self.assertEqual(filesystem_file_identity(lock_path), filesystem_file_identity(out / ".mtg-kernel-train.lock"))
            finally:
                release_marker.write_text("release", encoding="utf-8")
                owner_code = owner.wait(timeout=30)
            self.assertEqual(owner_code, 0)
            train(
                env_bin=launcher,
                out_dir=out,
                base_seed=71501,
                until_update=0,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            self.assertEqual(read_json_file(out / "latest.json")["update"], 0)

    def test_physical_alias_concurrent_trainers_exclude_loser_without_second_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            real_parent = tmp / "real_parent"
            real_parent.mkdir()
            out = real_parent / "run"
            launcher = fake_launcher(tmp, "train_pair_slow")
            train(
                env_bin=launcher,
                out_dir=out,
                base_seed=71501,
                until_update=0,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            if os.name == "nt":
                alias_parent = tmp / "alias_parent"
                completed = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(alias_parent), str(real_parent)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if completed.returncode != 0:
                    self.fail("Windows junction coverage could not create mklink /J")
            else:
                alias_parent = tmp / "alias_parent"
                try:
                    os.symlink(real_parent, alias_parent, target_is_directory=True)
                except (OSError, NotImplementedError) as exc:
                    raise unittest.SkipTest(f"POSIX symlink ancestor unavailable: {exc}") from exc
            alias_out = alias_parent / "run"
            env = _subprocess_env()

            def args(root: Path) -> list[str]:
                return [
                    sys.executable,
                    "-m",
                    "mtg_kernel_rl",
                    "train",
                    "--env-bin",
                    str(launcher),
                    "--out-dir",
                    str(root),
                    "--resume",
                    str(root / "latest.json"),
                    "--until-update",
                    "1",
                ]

            owner = subprocess.Popen(args(out), cwd=Path.cwd(), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.25)
            loser = subprocess.run(args(alias_out), cwd=Path.cwd(), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            owner_code = owner.wait(timeout=30)
            self.assertEqual(owner_code, 0)
            self.assertNotEqual(loser.returncode, 0)
            self.assertEqual(read_json_file(out / "latest.json")["update"], 1)
            self.assertEqual(sorted(p.name for p in (out / "updates").glob("update-*.json")), ["update-00000000.json", "update-00000001.json"])

    def test_direct_output_root_link_or_reparse_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "train_pair")
            outside = tmp / "outside"
            outside.mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_bytes(b"unchanged")
            root_link = tmp / "root_link"
            if os.name == "nt":
                completed = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(root_link), str(outside)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if completed.returncode != 0:
                    self.fail("Windows root junction coverage could not create mklink /J")
            else:
                try:
                    os.symlink(outside, root_link, target_is_directory=True)
                except (OSError, NotImplementedError) as exc:
                    raise unittest.SkipTest(f"POSIX root symlink unavailable: {exc}") from exc
            with self.assertRaises(ValueError):
                train(
                    env_bin=launcher,
                    out_dir=root_link,
                    base_seed=71501,
                    until_update=0,
                    batch_episodes=2,
                    learning_rate=0.001,
                    value_coef=0.5,
                    max_decisions=8,
                )
            self.assertEqual(sentinel.read_bytes(), b"unchanged")


if __name__ == "__main__":
    unittest.main()
