from __future__ import annotations

import copy
import dataclasses
import hashlib
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from mtg_kernel_rl.artifacts import read_json_file, write_json_atomic
from mtg_kernel_rl.checkpoint import load_checkpoint_file, save_checkpoint_file
from mtg_kernel_rl.training_store import TrainingStore
import mtg_kernel_rl.artifact_io as artifact_io
import mtg_kernel_rl.checkpoint_io as checkpoint_io
import mtg_kernel_rl.training_store as store_mod
import mtg_kernel_rl.trainer as trainer_mod
from mtg_kernel_rl.trainer import train

from fixtures import fake_launcher


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(["kernel/python", "kernel/python/tests"])
    return env


def _train_fixture(root: Path, *, until_update: int = 4, base_seed: int = 71501) -> Path:
    launcher = fake_launcher(root, f"training_store_{until_update}_{base_seed}")
    out = root / f"run_{until_update}_{base_seed}"
    train(
        env_bin=launcher,
        out_dir=out,
        base_seed=base_seed,
        until_update=until_update,
        batch_episodes=2,
        learning_rate=0.001,
        value_coef=0.5,
        max_decisions=8,
    )
    return out


def _pin_latest(root: Path, update: int) -> None:
    sidecar = read_json_file(root / "checkpoints" / f"update-{update:08d}.json")
    write_json_atomic(root / "latest.json", {"schema": "kernel_rl_train_latest/v2", "update": update, "run_digest": sidecar["run_digest"], "head": sidecar["head"]})


def _latest_for(root: Path, update: int) -> dict[str, Any]:
    sidecar = read_json_file(root / "checkpoints" / f"update-{update:08d}.json")
    return {"schema": "kernel_rl_train_latest/v2", "update": update, "run_digest": sidecar["run_digest"], "head": sidecar["head"]}


def _tree_snapshot(root: Path) -> dict[str, tuple[str, int, str | None, int, int, int, int]]:
    out: dict[str, tuple[str, int, str | None, int, int, int, int]] = {}
    for path in sorted([root, *root.rglob("*")], key=lambda item: str(item.relative_to(root))):
        rel = str(path.relative_to(root))
        st = path.lstat()
        if path.is_symlink():
            kind = "link"
            digest = None
            size = 0
        elif path.is_dir():
            kind = "dir"
            digest = None
            size = 0
        else:
            kind = "file"
            if path.name == ".mtg-kernel-train.lock":
                digest = None
                size = st.st_size
            else:
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                size = len(data)
        out[rel] = (
            kind,
            size,
            digest,
            int(getattr(st, "st_dev", 0)),
            int(getattr(st, "st_ino", 0)),
            int(getattr(st, "st_nlink", 1)),
            int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
        )
    return out


class TrainingStoreTest(unittest.TestCase):
    def test_trainer_imports_canonical_training_store_contract_owner(self) -> None:
        self.assertIs(trainer_mod._validate_run_manifest, store_mod._validate_run_manifest)
        self.assertIs(trainer_mod._assert_run_matches_options, store_mod._assert_run_matches_options)
        self.assertIs(trainer_mod._validate_update_record, store_mod._validate_update_record)
        self.assertIs(trainer_mod._strict_validate_checkpoint_for_model, store_mod._strict_validate_checkpoint_for_model)
        self.assertIs(trainer_mod._validate_generation_bundle, store_mod._validate_generation_bundle)
        self.assertIs(trainer_mod._update_record_zero, store_mod._update_record_zero)

        code = (
            "import sys\n"
            "import mtg_kernel_rl.training_store as store\n"
            "assert 'mtg_kernel_rl.trainer' not in sys.modules, sorted(sys.modules)\n"
            "assert store.TrainingStore is not None\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path.cwd(),
            env=_subprocess_env(),
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_run_manifest_rejects_legacy_v3_protocol_and_schema_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            out = _train_fixture(Path(tmp_name), until_update=0)
            run = read_json_file(out / "run.json")
            for key in ("protocol_version", "schema_version"):
                legacy = copy.deepcopy(run)
                legacy["protocol_provenance"][key] = 3
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, rf"{key} mismatch"):
                    store_mod._validate_run_manifest(legacy)

    def test_validate_latest_read_counts_are_exact_linear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            source = _train_fixture(tmp, until_update=32)
            for head in (1, 4, 16, 32):
                target = tmp / f"head_{head}"
                shutil.copytree(source, target)
                _pin_latest(target, head)
                counts = {"total": 0, "updates": 0}
                original_json_read = artifact_io.read_regular_file_bytes
                original_checkpoint_read = checkpoint_io.read_regular_file_bytes

                def counted(path: str | Path, *args: Any, **kwargs: Any) -> Any:
                    result = original_json_read(path, *args, **kwargs)
                    text = str(path).replace("\\", "/")
                    counts["total"] += 1
                    if "/updates/update-" in text and text.endswith(".json"):
                        counts["updates"] += 1
                    return result

                def counted_checkpoint(path: str | Path, *args: Any, **kwargs: Any) -> Any:
                    result = original_checkpoint_read(path, *args, **kwargs)
                    counts["total"] += 1
                    return result

                artifact_io.read_regular_file_bytes = counted
                checkpoint_io.read_regular_file_bytes = counted_checkpoint
                try:
                    chain = TrainingStore(target).validate_latest()
                finally:
                    artifact_io.read_regular_file_bytes = original_json_read
                    checkpoint_io.read_regular_file_bytes = original_checkpoint_read
                n = head + 1
                self.assertEqual(chain.head.update, head)
                self.assertEqual(chain.read_counts.total, 2 + 3 * n)
                self.assertEqual(chain.read_counts.updates, n)
                self.assertEqual(counts["total"], 2 + 3 * n)
                self.assertEqual(counts["updates"], n)
            self.assertEqual((2 + 3 * 33, 33), (101, 33))

    def test_run_digest_comes_from_same_capture_that_was_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = _train_fixture(tmp, until_update=1)
            original = store_mod.read_authoritative_json_capture
            raced = {"done": False}

            def capture_then_corrupt(path: str | Path, kind: str) -> Any:
                captured = original(path, kind)
                if kind == "run" and not raced["done"]:
                    raced["done"] = True
                    bad = copy.deepcopy(captured.value)
                    bad["schema"] = "kernel_rl_train_run/v10"
                    write_json_atomic(path, bad)
                return captured

            store_mod.read_authoritative_json_capture = capture_then_corrupt
            try:
                chain = TrainingStore(out).validate_latest()
            finally:
                store_mod.read_authoritative_json_capture = original
            self.assertTrue(raced["done"])
            self.assertEqual(chain.head.update, 1)

    def test_policy_and_resume_load_second_reads_only_selected_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = _train_fixture(tmp, until_update=4)
            chain = TrainingStore(out).validate_latest()

            def counted_load(call: Any) -> list[str]:
                reads: list[str] = []
                original_json_read = artifact_io.read_regular_file_bytes
                original_checkpoint_read = checkpoint_io.read_regular_file_bytes

                def counted_json(path: str | Path, *args: Any, **kwargs: Any) -> Any:
                    reads.append(str(Path(path).relative_to(out)).replace("\\", "/"))
                    return original_json_read(path, *args, **kwargs)

                def counted_checkpoint(path: str | Path, *args: Any, **kwargs: Any) -> Any:
                    reads.append(str(Path(path).relative_to(out)).replace("\\", "/"))
                    return original_checkpoint_read(path, *args, **kwargs)

                artifact_io.read_regular_file_bytes = counted_json
                checkpoint_io.read_regular_file_bytes = counted_checkpoint
                try:
                    call()
                finally:
                    artifact_io.read_regular_file_bytes = original_json_read
                    checkpoint_io.read_regular_file_bytes = original_checkpoint_read
                return reads

            policy_reads = counted_load(lambda: chain.load_policy(chain.snapshots[2]))
            self.assertEqual(
                sorted(policy_reads),
                [
                    "checkpoints/update-00000002.json",
                    "checkpoints/update-00000002.pt",
                    "updates/update-00000002.json",
                ],
            )
            resume_reads = counted_load(lambda: chain.load_resume(chain.head))
            self.assertEqual(
                sorted(resume_reads),
                [
                    "checkpoints/update-00000004.json",
                    "checkpoints/update-00000004.pt",
                    "updates/update-00000004.json",
                ],
            )
            forbidden = {"run.json", "latest.json", "episodes.jsonl", "updates.jsonl", "summary.json"}
            self.assertTrue(forbidden.isdisjoint(policy_reads))
            self.assertTrue(forbidden.isdisjoint(resume_reads))

    def test_latest_capture_pins_old_head_and_ignores_later_debris(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = _train_fixture(tmp, until_update=8)
            _pin_latest(out, 4)
            latest8 = _latest_for(out, 8)
            original = store_mod.read_authoritative_json_capture
            advanced = {"done": False}

            def advance_after_latest(path: str | Path, kind: str) -> Any:
                captured = original(path, kind)
                if kind == "latest" and not advanced["done"]:
                    advanced["done"] = True
                    write_json_atomic(path, latest8)
                    (out / "updates" / "update-99999999.json").write_text("not canonical debris\n", encoding="utf-8")
                    (out / "summary.json").write_text("not json cache debris\n", encoding="utf-8")
                return captured

            store_mod.read_authoritative_json_capture = advance_after_latest
            try:
                chain = TrainingStore(out).validate_latest()
            finally:
                store_mod.read_authoritative_json_capture = original
            self.assertTrue(advanced["done"])
            self.assertEqual(chain.head.update, 4)
            self.assertEqual(chain.latest_record["update"], 4)

    def test_pinned_older_head_load_remains_valid_after_latest_advances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = _train_fixture(tmp, until_update=5)
            _pin_latest(out, 2)
            chain = TrainingStore(out).validate_latest()
            self.assertEqual(chain.head.update, 2)

            write_json_atomic(out / "latest.json", _latest_for(out, 5))
            (out / "updates" / "update-99999999.json").write_text("later debris\n", encoding="utf-8")
            (out / "summary.json").write_text("stale cache after pin\n", encoding="utf-8")

            policy = chain.load_policy(chain.head)
            resume = chain.load_resume(chain.head)
            self.assertEqual(policy.ref.update, 2)
            self.assertEqual(resume.completed_update, 2)
            self.assertTrue(all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in policy.model.state_dict().values()))
            self.assertTrue(all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in resume.model.state_dict().values()))

    def test_loads_fail_closed_when_pinned_artifacts_are_replaced_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            source = _train_fixture(tmp, until_update=2)

            def replace_update(root: Path) -> None:
                shutil.copy2(root / "updates" / "update-00000000.json", root / "updates" / "update-00000002.json")

            def replace_sidecar(root: Path) -> None:
                shutil.copy2(root / "checkpoints" / "update-00000000.json", root / "checkpoints" / "update-00000002.json")

            def replace_checkpoint(root: Path) -> None:
                shutil.copy2(root / "checkpoints" / "update-00000000.pt", root / "checkpoints" / "update-00000002.pt")

            for name, mutator in {
                "update": replace_update,
                "sidecar": replace_sidecar,
                "checkpoint": replace_checkpoint,
            }.items():
                with self.subTest(name=name):
                    target = tmp / f"tamper_{name}"
                    shutil.copytree(source, target)
                    chain = TrainingStore(target).validate_latest()
                    mutator(target)
                    with self.assertRaises(ValueError):
                        chain.load_policy(chain.head)
                    with self.assertRaises(ValueError):
                        chain.load_resume(chain.head)

    def test_public_reader_preserves_process_global_torch_state_in_fresh_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = _train_fixture(tmp, until_update=2)
            script = tmp / "reader_state_probe.py"
            script.write_text(
                r'''
import random
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])

torch.set_default_dtype(torch.float64)
torch.set_default_device("meta")
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.use_deterministic_algorithms(False)
torch.set_float32_matmul_precision("medium")
random.seed(1234567)
np.random.seed(2345678)
torch.manual_seed(3456789)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(4567890)

from mtg_kernel_rl.training_store import TrainingStore
import mtg_kernel_rl.determinism as determinism


def snapshot():
    numpy_state = np.random.get_state()
    return {
        "default_dtype": str(torch.get_default_dtype()),
        "default_device": str(torch.get_default_device()),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "configured": determinism._TORCH_CONFIGURED,
        "python_rng": random.getstate(),
        "numpy_rng": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu_rng": torch.random.get_rng_state().clone(),
        "torch_cuda_rng": [item.clone() for item in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def assert_same(before, label):
    after = snapshot()
    for key in (
        "default_dtype",
        "default_device",
        "num_threads",
        "num_interop_threads",
        "deterministic",
        "warn_only",
        "matmul_precision",
        "configured",
        "python_rng",
    ):
        assert after[key] == before[key], (label, key, before[key], after[key])
    assert after["numpy_rng"][0] == before["numpy_rng"][0], label
    assert np.array_equal(after["numpy_rng"][1], before["numpy_rng"][1]), label
    assert after["numpy_rng"][2:] == before["numpy_rng"][2:], label
    assert torch.equal(after["torch_cpu_rng"], before["torch_cpu_rng"]), label
    assert len(after["torch_cuda_rng"]) == len(before["torch_cuda_rng"]), label
    for left, right in zip(after["torch_cuda_rng"], before["torch_cuda_rng"]):
        assert torch.equal(left, right), label


def assert_cpu_float32(model):
    for tensor in model.state_dict().values():
        assert tensor.device.type == "cpu", tensor.device
        assert tensor.dtype == torch.float32, tensor.dtype


before = snapshot()
chain = TrainingStore(root).validate_latest()
assert_same(before, "validate_latest")
policy = chain.load_policy()
assert_cpu_float32(policy.model)
assert_same(before, "load_policy")
resume = chain.load_resume()
assert_cpu_float32(resume.model)
assert_same(before, "load_resume")
chain.head.update_path.write_text("{}\n", encoding="utf-8")
try:
    chain.load_policy(chain.head)
except Exception:
    pass
else:
    raise AssertionError("tampered load unexpectedly succeeded")
assert_same(before, "tampered_load")
''',
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(script), str(out)],
                cwd=Path.cwd(),
                env=_subprocess_env(),
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_trainer_resume_races_fail_before_recovery_cache_mutation_or_env_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "resume_race_launcher")
            source = tmp / "resume_race_source"
            train(
                env_bin=launcher,
                out_dir=source,
                base_seed=71501,
                until_update=2,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )

            def replace_update(root: Path) -> None:
                shutil.copy2(root / "updates" / "update-00000000.json", root / "updates" / "update-00000002.json")

            def replace_sidecar(root: Path) -> None:
                shutil.copy2(root / "checkpoints" / "update-00000000.json", root / "checkpoints" / "update-00000002.json")

            def replace_checkpoint(root: Path) -> None:
                shutil.copy2(root / "checkpoints" / "update-00000000.pt", root / "checkpoints" / "update-00000002.pt")

            original_store = trainer_mod.TrainingStore
            original_client = trainer_mod.KernelRlClient
            for name, mutator in {
                "update": replace_update,
                "sidecar": replace_sidecar,
                "checkpoint": replace_checkpoint,
            }.items():
                with self.subTest(name=name):
                    target = tmp / f"resume_race_{name}"
                    shutil.copytree(source, target)
                    (target / "episodes.jsonl").write_text("stale cache\n", encoding="utf-8")
                    post_race: dict[str, Any] = {}
                    launched = {"value": False}

                    class RacingStore:
                        def __init__(self, root: str | Path):
                            self._inner = original_store(root)

                        def validate_latest(self) -> Any:
                            chain = self._inner.validate_latest()
                            mutator(target)
                            post_race["tree"] = _tree_snapshot(target)
                            return chain

                    class NoEnvClient:
                        def __init__(self, *args: Any, **kwargs: Any) -> None:
                            launched["value"] = True
                            raise AssertionError("environment launched before immutable resume load failed")

                    trainer_mod.TrainingStore = RacingStore
                    trainer_mod.KernelRlClient = NoEnvClient
                    try:
                        with self.assertRaises(ValueError):
                            trainer_mod.train(env_bin=launcher, out_dir=target, resume=target / "latest.json", until_update=3)
                    finally:
                        trainer_mod.TrainingStore = original_store
                        trainer_mod.KernelRlClient = original_client
                    self.assertFalse(launched["value"])
                    self.assertIn("tree", post_race)
                    self.assertEqual(_tree_snapshot(target), post_race["tree"])

    def test_until_update_before_committed_fails_before_mutation_rng_or_env_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "invalid_target_launcher")
            out = tmp / "invalid_target_run"
            train(
                env_bin=launcher,
                out_dir=out,
                base_seed=71501,
                until_update=2,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=8,
            )
            (out / "episodes.jsonl").write_text("stale cache\n", encoding="utf-8")
            (out / "updates.jsonl").write_text("stale cache\n", encoding="utf-8")
            (out / "summary.json").write_text("stale cache\n", encoding="utf-8")
            before_tree = _tree_snapshot(out)
            random.seed(110011)
            torch.manual_seed(220022)
            py_state = random.getstate()
            torch_state = torch.random.get_rng_state().clone()
            original_client = trainer_mod.KernelRlClient
            launched = {"value": False}

            class NoEnvClient:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    launched["value"] = True
                    raise AssertionError("environment launched for invalid resume target")

            trainer_mod.KernelRlClient = NoEnvClient
            try:
                with self.assertRaises(ValueError):
                    trainer_mod.train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1)
            finally:
                trainer_mod.KernelRlClient = original_client
            self.assertFalse(launched["value"])
            self.assertEqual(_tree_snapshot(out), before_tree)
            self.assertEqual(random.getstate(), py_state)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_noop_resume_repairs_caches_only_after_immutable_load_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "noop_repair_launcher")
            out = tmp / "noop_repair_run"
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
            for name in ("episodes.jsonl", "updates.jsonl", "summary.json"):
                (out / name).unlink()
            original_store = trainer_mod.TrainingStore
            observed = {"load_called": False}
            test_case = self

            class CheckingChain:
                def __init__(self, chain: Any):
                    self._chain = chain

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._chain, name)

                def load_resume(self, ref: Any | None = None) -> Any:
                    observed["load_called"] = True
                    self_ref = self._chain.head if ref is None else ref
                    test_case.assertFalse((out / "episodes.jsonl").exists())
                    test_case.assertFalse((out / "updates.jsonl").exists())
                    test_case.assertFalse((out / "summary.json").exists())
                    return self._chain.load_resume(self_ref)

            class CheckingStore:
                def __init__(self, root: str | Path):
                    self._inner = original_store(root)

                def validate_latest(self) -> Any:
                    return CheckingChain(self._inner.validate_latest())

            trainer_mod.TrainingStore = CheckingStore
            try:
                result = trainer_mod.train(env_bin=launcher, out_dir=out, resume=out / "latest.json", until_update=1)
            finally:
                trainer_mod.TrainingStore = original_store
            self.assertTrue(observed["load_called"])
            self.assertEqual(result["completed_update"], 1)
            self.assertTrue((out / "episodes.jsonl").is_file())
            self.assertTrue((out / "updates.jsonl").is_file())
            self.assertTrue((out / "summary.json").is_file())

    def test_policy_and_resume_loads_are_detached_rng_preserving_and_non_aliasing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = _train_fixture(tmp, until_update=4)
            chain = TrainingStore(out).validate_latest()

            py_state = random.getstate()
            torch_state = torch.random.get_rng_state().clone()
            policy = chain.load_policy(chain.snapshots[1])
            self.assertIs(policy.ref, chain.snapshots[1])
            self.assertFalse(policy.model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in policy.model.parameters()))
            self.assertTrue(all(tensor.device.type == "cpu" for tensor in policy.model.state_dict().values()))
            self.assertFalse(hasattr(policy, "optimizer"))
            self.assertEqual(random.getstate(), py_state)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
            first_name = next(iter(policy.model.state_dict()))
            with torch.no_grad():
                policy.model.state_dict()[first_name].reshape(-1)[0] += 1
            policy_again = chain.load_policy(chain.snapshots[1])
            self.assertFalse(torch.equal(policy.model.state_dict()[first_name], policy_again.model.state_dict()[first_name]))

            with self.assertRaises(ValueError):
                chain.load_resume(chain.snapshots[1])
            resume_a = chain.load_resume(chain.head)
            self.assertTrue(resume_a.model.training)
            self.assertIsInstance(resume_a.optimizer, torch.optim.Adam)
            self.assertEqual(resume_a.completed_update, 4)
            self.assertEqual(random.getstate(), py_state)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
            resume_a.outcomes_by_learner_seat["p0"]["win"] = 999
            resume_a.learner_decisions_by_seat["p0"] = 999
            resume_a.torch_cpu_rng_state[0] ^= 1
            first_payload_key = next(iter(resume_a.checkpoint_payload["model_state"]))
            resume_a.checkpoint_payload["model_state"][first_payload_key].reshape(-1)[0] += 1
            resume_b = chain.load_resume(chain.head)
            self.assertIsNot(resume_a.model, resume_b.model)
            self.assertIsNot(resume_a.optimizer, resume_b.optimizer)
            self.assertIsNot(resume_a.checkpoint_payload, resume_b.checkpoint_payload)
            for key, tensor_a in resume_a.model.state_dict().items():
                tensor_b = resume_b.model.state_dict()[key]
                self.assertIsNot(tensor_a, tensor_b)
                if tensor_a.numel() > 0:
                    self.assertNotEqual(tensor_a.untyped_storage().data_ptr(), tensor_b.untyped_storage().data_ptr(), key)
            params_a = list(resume_a.model.parameters())
            params_b = list(resume_b.model.parameters())
            self.assertEqual([id(param) for param in resume_a.optimizer.param_groups[0]["params"]], [id(param) for param in params_a])
            self.assertEqual([id(param) for param in resume_b.optimizer.param_groups[0]["params"]], [id(param) for param in params_b])
            self.assertTrue(set(resume_a.optimizer.state).issubset(set(params_a)))
            self.assertTrue(set(resume_b.optimizer.state).issubset(set(params_b)))
            for param_a, param_b in zip(params_a, params_b):
                self.assertIsNot(param_a, param_b)
                slot_a = resume_a.optimizer.state.get(param_a, {})
                slot_b = resume_b.optimizer.state.get(param_b, {})
                self.assertEqual(set(slot_a), set(slot_b))
                for key in slot_a:
                    value_a = slot_a[key]
                    value_b = slot_b[key]
                    self.assertIsNot(value_a, value_b)
                    if value_a.numel() > 0:
                        self.assertNotEqual(value_a.untyped_storage().data_ptr(), value_b.untyped_storage().data_ptr(), key)

            def tensor_ptrs(value: Any) -> set[int]:
                if isinstance(value, torch.Tensor):
                    return set() if value.numel() == 0 else {value.untyped_storage().data_ptr()}
                if isinstance(value, dict):
                    out: set[int] = set()
                    for item in value.values():
                        out.update(tensor_ptrs(item))
                    return out
                if isinstance(value, (list, tuple)):
                    out: set[int] = set()
                    for item in value:
                        out.update(tensor_ptrs(item))
                    return out
                return set()

            payload_ptrs_a = tensor_ptrs(resume_a.checkpoint_payload)
            payload_ptrs_b = tensor_ptrs(resume_b.checkpoint_payload)
            model_ptrs_a = tensor_ptrs(resume_a.model.state_dict())
            optimizer_ptrs_a = tensor_ptrs({str(index): slot for index, slot in enumerate(resume_a.optimizer.state.values())})
            self.assertTrue(payload_ptrs_a.isdisjoint(payload_ptrs_b))
            self.assertTrue(payload_ptrs_a.isdisjoint(model_ptrs_a))
            self.assertTrue(payload_ptrs_a.isdisjoint(optimizer_ptrs_a))
            self.assertNotEqual(resume_b.outcomes_by_learner_seat["p0"]["win"], 999)
            self.assertNotEqual(resume_b.learner_decisions_by_seat["p0"], 999)
            self.assertFalse(torch.equal(resume_a.torch_cpu_rng_state, resume_b.torch_cpu_rng_state))
            self.assertFalse(
                torch.equal(
                    resume_a.checkpoint_payload["model_state"][first_payload_key],
                    resume_b.checkpoint_payload["model_state"][first_payload_key],
                )
            )
            resume_c = chain.load_resume(chain.head)
            self.assertTrue(payload_ptrs_b.isdisjoint(tensor_ptrs(resume_c.checkpoint_payload)))

    def test_cross_chain_modified_and_forged_refs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            left = _train_fixture(tmp, until_update=2, base_seed=71501)
            right = _train_fixture(tmp, until_update=2, base_seed=91501)
            left_chain = TrainingStore(left).validate_latest()
            right_chain = TrainingStore(right).validate_latest()
            cases = [
                right_chain.head,
                dataclasses.replace(left_chain.head, root=right_chain.head.root),
                dataclasses.replace(left_chain.head, update_path=left_chain.head.sidecar_path),
                dataclasses.replace(left_chain.head, checkpoint_sha256="0" * 64),
                dataclasses.replace(left_chain.head),
            ]
            for ref in cases:
                with self.subTest(ref=ref):
                    with self.assertRaises(ValueError):
                        left_chain.load_policy(ref)

    def test_schema_corruption_fails_without_tree_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            source = _train_fixture(tmp, until_update=1)

            def case(name: str, mutator: Any) -> None:
                target = tmp / name
                shutil.copytree(source, target)
                mutator(target)
                before = _tree_snapshot(target)
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        TrainingStore(target).validate_latest()
                    self.assertEqual(_tree_snapshot(target), before)

            case("old_run_schema", lambda p: write_json_atomic(p / "run.json", {**read_json_file(p / "run.json"), "schema": "kernel_rl_train_run/v10"}))
            case(
                "run_deck_id_drift",
                lambda p: (
                    (lambda run: (
                        run["environment"]["deck_ids"].__setitem__(1, "Rally"),
                        write_json_atomic(p / "run.json", run),
                    ))(read_json_file(p / "run.json"))
                ),
            )
            case(
                "run_deck_hash_shape",
                lambda p: (
                    (lambda run: (
                        run["environment"].__setitem__("deck_hashes", [1]),
                        write_json_atomic(p / "run.json", run),
                    ))(read_json_file(p / "run.json"))
                ),
            )
            case(
                "old_boundary_schema",
                lambda p: write_json_atomic(
                    p / "run.json",
                    {
                        **read_json_file(p / "run.json"),
                        "artifact_boundary": {
                            **read_json_file(p / "run.json")["artifact_boundary"],
                            "schema": "kernel_rl_artifact_boundary/v8",
                        },
                    },
                ),
            )
            case("latest_extra_field", lambda p: write_json_atomic(p / "latest.json", {**read_json_file(p / "latest.json"), "extra": True}))
            case(
                "checkpoint_counter_drift",
                lambda p: (
                    (lambda payload: (
                        payload.__setitem__("next_episode", payload["next_episode"] + 2),
                        save_checkpoint_file(p / "checkpoints" / "update-00000001.pt", payload),
                    ))(load_checkpoint_file(p / "checkpoints" / "update-00000001.pt"))
                ),
            )

    def test_public_reader_does_not_read_or_repair_derived_caches_and_does_not_import_trainer(self) -> None:
        source_text = Path(store_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from .trainer", source_text)
        self.assertNotIn("KernelRlClient", source_text)
        self.assertNotIn("OutputLock", source_text)
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = _train_fixture(tmp, until_update=1)
            (out / "episodes.jsonl").write_text("absolute path cache debris /tmp/forbidden\n", encoding="utf-8")
            (out / "updates.jsonl").write_text("not json\n", encoding="utf-8")
            (out / "summary.json").write_text("not json\n", encoding="utf-8")
            before = _tree_snapshot(out)
            chain = TrainingStore(out).validate_latest()
            self.assertEqual(chain.head.update, 1)
            self.assertEqual(_tree_snapshot(out), before)

    def test_public_mutable_aliases_cannot_affect_later_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            out = _train_fixture(tmp, until_update=2)
            chain = TrainingStore(out).validate_latest()
            run = chain.run_record
            latest = chain.latest_record
            records = chain.update_records
            run["trainer"]["base_seed"] = 1
            latest["update"] = 0
            records[1]["episode_summaries"].clear()
            self.assertNotEqual(chain.run_record["trainer"]["base_seed"], 1)
            self.assertEqual(chain.latest_record["update"], 2)
            self.assertTrue(chain.update_records[1]["episode_summaries"])
            resume = chain.load_resume()
            self.assertEqual(resume.completed_update, 2)


if __name__ == "__main__":
    unittest.main()
