from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import mtg_kernel_rl.cli as cli_mod
import mtg_kernel_rl.evaluator as evaluator_mod
from mtg_kernel_rl.artifact_io import canonical_json_bytes, read_json_file, sha256_bytes, validate_training_json_privacy
from mtg_kernel_rl.artifacts import set_fault_injector
from mtg_kernel_rl.client import Decision
from mtg_kernel_rl.determinism import derive_evaluation_env_seed
from mtg_kernel_rl.evaluation_store import validate_evaluation
from mtg_kernel_rl.evaluator import EvaluationResult, _select_greedy_action, _validate_request, evaluate
from mtg_kernel_rl.output_lock import OutputLock
from mtg_kernel_rl.path_safety import OUTPUT_LOCK_FILE_NAME
from mtg_kernel_rl.trainer import train
from mtg_kernel_rl.training_store import TrainingStore

from fixtures import PROVENANCE, actor_observation, fake_launcher, legal_actions


def _train_fixture(
    root: Path,
    *,
    scenario: str = "valid",
    until_update: int = 0,
    max_decisions: int = 8,
    extra_env: dict[str, str] | None = None,
) -> tuple[Path, Path, str]:
    launcher = fake_launcher(root, scenario, extra_env)
    store = root / "training-store"
    train(
        env_bin=launcher,
        out_dir=store,
        base_seed=71_501,
        until_update=until_update,
        batch_episodes=2,
        learning_rate=0.001,
        value_coef=0.5,
        max_decisions=max_decisions,
    )
    head = TrainingStore(store).validate_latest().head.head
    return launcher, store, head


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {name: (root / name).read_bytes() for name in ("games.jsonl", "pairs.jsonl", "run.json")}


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def _rewrite_manifest_file_metadata(root: Path, name: str) -> None:
    manifest = read_json_file(root / "run.json")
    data = (root / name).read_bytes()
    manifest["files"][name] = {
        "row_count": len(data.splitlines()),
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }
    (root / "run.json").write_bytes(canonical_json_bytes(manifest))


def _evaluate_fixture(root: Path, launcher: Path, store: Path, head: str, *, out_name: str = "evaluation") -> Path:
    out = root / out_name
    evaluate(
        training_store=store,
        expected_candidate_head=head,
        env_bin=launcher,
        out_dir=out,
        pairs=1,
        base_seed=4,
        bootstrap_replicates=1_000,
        max_decisions=8,
        timeout_ms=5_000,
    )
    return out


def _create_directory_alias(test: unittest.TestCase, alias: Path, target: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            test.fail(f"Windows junction creation failed: {result.stdout} {result.stderr}")
        return
    try:
        os.symlink(target, alias, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        raise unittest.SkipTest(f"directory symlink unavailable: {exc}") from exc


def _remove_directory_alias(alias: Path) -> None:
    if os.name == "nt":
        os.rmdir(alias)
    else:
        alias.unlink()


_HARD_EXIT_EVALUATOR = r"""
import os
import sys
from pathlib import Path

from mtg_kernel_rl.artifacts import set_fault_injector
from mtg_kernel_rl.evaluator import evaluate

store, head, launcher, out, wanted_boundary, wanted_name, marker = sys.argv[1:]
marker_path = Path(marker)

def injector(boundary, path):
    if boundary == wanted_boundary and path is not None and path.name == wanted_name:
        with marker_path.open("xb") as handle:
            handle.write(b"reached")
            handle.flush()
            os.fsync(handle.fileno())
        os._exit(91)

set_fault_injector(injector)
evaluate(
    training_store=Path(store),
    expected_candidate_head=head,
    env_bin=Path(launcher),
    out_dir=Path(out),
    pairs=1,
    base_seed=4,
    bootstrap_replicates=1000,
    max_decisions=8,
    timeout_ms=5000,
)
raise SystemExit(92)
"""


class EvaluatorEndToEndTest(unittest.TestCase):
    def test_aa_control_is_half_score_deterministic_private_and_rng_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            before_store = _tree_bytes(store)
            random.seed(123_456)
            torch.manual_seed(789)
            python_rng = random.getstate()
            torch_rng = torch.get_rng_state().clone()

            first = evaluate(
                training_store=store,
                expected_candidate_head=head,
                env_bin=launcher,
                out_dir=root / "eval-a",
                pairs=3,
                base_seed=42,
                bootstrap_replicates=1_000,
                max_decisions=8,
                timeout_ms=5_000,
            )
            second = evaluate(
                training_store=store,
                expected_candidate_head=head,
                env_bin=launcher,
                out_dir=root / "eval-b",
                pairs=3,
                base_seed=42,
                bootstrap_replicates=1_000,
                max_decisions=8,
                timeout_ms=5_000,
            )
            self.assertEqual(first, second)
            self.assertEqual((first.total_half_points, first.estimate), (6, 0.5))
            self.assertEqual(random.getstate(), python_rng)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            self.assertEqual(_tree_bytes(store), before_store)
            self.assertEqual(_artifact_bytes(root / "eval-a"), _artifact_bytes(root / "eval-b"))

            pairs = _jsonl(root / "eval-a" / "pairs.jsonl")
            self.assertEqual([row["total_half_points"] for row in pairs], [2, 2, 2])
            manifest = read_json_file(root / "eval-a" / "run.json")
            paired = manifest["statistics"]["paired"]
            self.assertEqual(paired["estimate_hex"], (0.5).hex())
            self.assertEqual((paired["bootstrap"]["lower_hex"], paired["bootstrap"]["upper_hex"]), ((0.5).hex(), (0.5).hex()))
            self.assertEqual(
                (paired["sign_test"]["ties"], paired["sign_test"]["p_value_numerator_hex"], paired["sign_test"]["p_value_denominator_hex"]),
                (3, "0x1", "0x1"),
            )
            validate_training_json_privacy(manifest)
            combined = b"".join(_artifact_bytes(root / "eval-a").values()).decode("ascii")
            for forbidden in (str(root), '"stable_id":', '"card_name":', '"legal_actions":', '"observation":'):
                self.assertNotIn(forbidden, combined)
            self.assertEqual(validate_evaluation(root / "eval-a").run_sha256, first.run_sha256)

    def test_reset_schedule_seat_swap_and_actor_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            request_log = root / "requests.jsonl"
            launcher, store, head = _train_fixture(
                root,
                scenario="train_pair",
                extra_env={"FAKE_REQUEST_LOG": str(request_log)},
            )
            request_log.unlink(missing_ok=True)
            result = evaluate(
                training_store=store,
                expected_candidate_head=head,
                env_bin=launcher,
                out_dir=root / "evaluation",
                pairs=3,
                base_seed=91,
                bootstrap_replicates=1_000,
                max_decisions=8,
                timeout_ms=5_000,
            )
            requests = _jsonl(request_log)
            resets = [row for row in requests if row["request_type"] == "reset"]
            self.assertEqual([row["episode_id"] for row in resets], list(range(6)))
            expected_seeds = [derive_evaluation_env_seed(91, pair) for pair in range(3) for _ in range(2)]
            self.assertEqual([row["env_seed"] for row in resets], expected_seeds)
            self.assertEqual([row["max_decisions"] for row in resets], [8] * 6)
            games = _jsonl(root / "evaluation" / "games.jsonl")
            self.assertEqual([row["candidate_seat"] for row in games], ["p0", "p1"] * 3)
            self.assertTrue(all(row["candidate_decisions"] == 1 and row["baseline_decisions"] == 1 for row in games))
            self.assertEqual([row["total_half_points"] for row in _jsonl(root / "evaluation" / "pairs.jsonl")], [4, 1, 1])
            self.assertEqual((result.total_half_points, result.estimate), (6, 0.5))

    def test_topology_head_and_environment_preflight_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            marker = root / "started.txt"
            launcher, store, head = _train_fixture(
                root,
                scenario="train_pair",
                until_update=1,
                extra_env={"FAKE_START_MARKER": str(marker)},
            )
            marker.unlink(missing_ok=True)
            with self.assertRaises(ValueError):
                evaluate(
                    training_store=store,
                    expected_candidate_head="0" * 64,
                    env_bin=launcher,
                    out_dir=root / "head-mismatch",
                    pairs=1,
                    base_seed=1,
                    bootstrap_replicates=1_000,
                    max_decisions=8,
                    timeout_ms=5_000,
                )
            self.assertFalse(marker.exists())
            self.assertFalse((root / "head-mismatch").exists())
            with self.assertRaises(ValueError):
                evaluate(
                    training_store=store,
                    expected_candidate_head=head,
                    env_bin=launcher,
                    out_dir=store / "nested-evaluation",
                    pairs=1,
                    base_seed=1,
                    bootstrap_replicates=1_000,
                    max_decisions=8,
                    timeout_ms=5_000,
                )
            self.assertFalse(marker.exists())
            self.assertFalse((store / "nested-evaluation").exists())

            other = fake_launcher(root, "valid", {"FAKE_START_MARKER": str(marker)})
            with self.assertRaises(ValueError):
                evaluate(
                    training_store=store,
                    expected_candidate_head=head,
                    env_bin=other,
                    out_dir=root / "env-mismatch",
                    pairs=1,
                    base_seed=1,
                    bootstrap_replicates=1_000,
                    max_decisions=8,
                    timeout_ms=5_000,
                )
            self.assertFalse(marker.exists())

            chain = TrainingStore(store).validate_latest()
            counting = mock.Mock()
            counting.validate_latest.return_value = chain
            with mock.patch.object(evaluator_mod, "TrainingStore", return_value=counting):
                evaluate(
                    training_store=store,
                    expected_candidate_head=head,
                    env_bin=launcher,
                    out_dir=root / "success",
                    pairs=1,
                    base_seed=1,
                    bootstrap_replicates=1_000,
                    max_decisions=8,
                    timeout_ms=5_000,
                )
            self.assertEqual(counting.validate_latest.call_count, 1)
            manifest = read_json_file(root / "success" / "run.json")
            self.assertEqual([row["update"] for row in manifest["snapshots"]], [1, 0])
            self.assertEqual([row["role"] for row in manifest["snapshots"]], ["candidate", "baseline"])

    def test_runtime_failure_publishes_nothing_and_lock_only_retry_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            scenario_file = root / "scenario.txt"
            scenario_file.write_text("valid", encoding="utf-8")
            launcher, store, head = _train_fixture(
                root,
                scenario="switchable",
                extra_env={"FAKE_SCENARIO_FILE": str(scenario_file)},
            )
            scenario_file.write_text("train_late_fault", encoding="utf-8")
            out = root / "evaluation"
            with self.assertRaises(Exception):
                evaluate(
                    training_store=store,
                    expected_candidate_head=head,
                    env_bin=launcher,
                    out_dir=out,
                    pairs=1,
                    base_seed=3,
                    bootstrap_replicates=1_000,
                    max_decisions=8,
                    timeout_ms=5_000,
                )
            self.assertEqual({path.name for path in out.iterdir()}, {OUTPUT_LOCK_FILE_NAME})
            scenario_file.write_text("valid", encoding="utf-8")
            result = evaluate(
                training_store=store,
                expected_candidate_head=head,
                env_bin=launcher,
                out_dir=out,
                pairs=1,
                base_seed=3,
                bootstrap_replicates=1_000,
                max_decisions=8,
                timeout_ms=5_000,
            )
            self.assertEqual(result.estimate, 0.5)
            self.assertEqual({path.name for path in out.iterdir()}, {OUTPUT_LOCK_FILE_NAME, "games.jsonl", "pairs.jsonl", "run.json"})

    def test_local_cap_rejects_continuation_and_accepts_terminal_exactly_at_cap(self) -> None:
        with tempfile.TemporaryDirectory() as valid_name, tempfile.TemporaryDirectory() as long_name:
            valid_root = Path(valid_name)
            launcher, store, head = _train_fixture(valid_root, max_decisions=1)
            result = evaluate(
                training_store=store,
                expected_candidate_head=head,
                env_bin=launcher,
                out_dir=valid_root / "evaluation",
                pairs=1,
                base_seed=9,
                bootstrap_replicates=1_000,
                max_decisions=1,
                timeout_ms=5_000,
            )
            self.assertEqual(result.game_count, 2)

            long_root = Path(long_name)
            launcher, store, head = _train_fixture(long_root, scenario="train_pair", max_decisions=1)
            out = long_root / "evaluation"
            with self.assertRaises(RuntimeError):
                evaluate(
                    training_store=store,
                    expected_candidate_head=head,
                    env_bin=launcher,
                    out_dir=out,
                    pairs=1,
                    base_seed=9,
                    bootstrap_replicates=1_000,
                    max_decisions=1,
                    timeout_ms=5_000,
                )
            self.assertEqual({path.name for path in out.iterdir()}, {OUTPUT_LOCK_FILE_NAME})


class EvaluationPublicationProofTest(unittest.TestCase):
    def test_data_file_faults_are_uncommitted_and_run_fault_is_postcommit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            cases = (
                ("games.jsonl", {OUTPUT_LOCK_FILE_NAME, "games.jsonl"}, False),
                ("pairs.jsonl", {OUTPUT_LOCK_FILE_NAME, "games.jsonl", "pairs.jsonl"}, False),
                ("run.json", {OUTPUT_LOCK_FILE_NAME, "games.jsonl", "pairs.jsonl", "run.json"}, True),
            )
            for target_name, expected_entries, committed in cases:
                with self.subTest(target=target_name):
                    out = root / f"fault-{target_name.replace('.', '-')}"
                    fired = {"value": False}

                    def injector(boundary: str, path: Path | None) -> None:
                        if (
                            not fired["value"]
                            and boundary == "json_replace_after"
                            and path is not None
                            and path.name == target_name
                        ):
                            fired["value"] = True
                            raise RuntimeError(f"fault after {target_name}")

                    previous = set_fault_injector(injector)
                    try:
                        with self.assertRaisesRegex(RuntimeError, target_name.replace(".", r"\.")):
                            _evaluate_fixture(root, launcher, store, head, out_name=out.name)
                    finally:
                        set_fault_injector(previous)
                    self.assertTrue(fired["value"])
                    self.assertEqual({path.name for path in out.iterdir()}, expected_entries)
                    if committed:
                        validated = validate_evaluation(out)
                        self.assertEqual((validated.total_half_points, validated.estimate), (2, 0.5))
                    else:
                        self.assertFalse((out / "run.json").exists())
                        with self.assertRaises(ValueError):
                            validate_evaluation(out)
                    with self.assertRaises(FileExistsError):
                        _evaluate_fixture(root, launcher, store, head, out_name=out.name)
                    self.assertEqual({path.name for path in out.iterdir()}, expected_entries)

    def test_hard_exit_at_every_atomic_replace_boundary_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            source_root = Path(__file__).resolve().parents[1]
            tests_root = Path(__file__).resolve().parent
            child_env = os.environ.copy()
            child_env["PYTHONPATH"] = os.pathsep.join((str(source_root), str(tests_root)))
            published_before = {
                "games.jsonl": set(),
                "pairs.jsonl": {"games.jsonl"},
                "run.json": {"games.jsonl", "pairs.jsonl"},
            }
            for boundary in ("json_replace_before", "json_replace_after"):
                for target_name in ("games.jsonl", "pairs.jsonl", "run.json"):
                    with self.subTest(boundary=boundary, target=target_name):
                        slug = f"{boundary}-{target_name.replace('.', '-')}"
                        out = root / slug
                        marker = root / f"{slug}.marker"
                        completed = subprocess.run(
                            [
                                sys.executable,
                                "-c",
                                _HARD_EXIT_EVALUATOR,
                                str(store),
                                head,
                                str(launcher),
                                str(out),
                                boundary,
                                target_name,
                                str(marker),
                            ],
                            cwd=Path(__file__).resolve().parents[3],
                            env=child_env,
                            capture_output=True,
                            text=True,
                            timeout=60,
                            check=False,
                        )
                        self.assertEqual(
                            completed.returncode,
                            91,
                            msg=f"stdout={completed.stdout!r} stderr={completed.stderr!r}",
                        )
                        self.assertEqual(marker.read_bytes(), b"reached")
                        self.assertNotIn(marker.name, {path.name for path in out.iterdir()})

                        names = {path.name for path in out.iterdir()}
                        expected = {OUTPUT_LOCK_FILE_NAME, *published_before[target_name]}
                        if boundary == "json_replace_after":
                            expected.add(target_name)
                            self.assertEqual(names, expected)
                        else:
                            temp_names = names - expected
                            self.assertEqual(len(temp_names), 1)
                            temp_name = next(iter(temp_names))
                            self.assertTrue(temp_name.startswith(f".{target_name}."), temp_name)
                            self.assertTrue(temp_name.endswith(".tmp"), temp_name)

                        # Process death must release the advisory lock even though
                        # the persistent lock file remains part of the output root.
                        with OutputLock(out):
                            pass

                        if boundary == "json_replace_after" and target_name == "run.json":
                            validated = validate_evaluation(out)
                            self.assertEqual((validated.total_half_points, validated.estimate), (2, 0.5))
                        else:
                            with self.assertRaises(ValueError):
                                validate_evaluation(out)
                        with self.assertRaises(FileExistsError):
                            _evaluate_fixture(root, launcher, store, head, out_name=out.name)
                        self.assertEqual({path.name for path in out.iterdir()}, names)


class EvaluationConcurrencyAndFailureProofTest(unittest.TestCase):
    def test_concurrent_latest_advance_preserves_selected_update_zero_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root, scenario="train_pair")
            original_run_game = evaluator_mod._run_game
            advanced = {"value": False}

            def advancing_run_game(*args, **kwargs):  # type: ignore[no-untyped-def]
                row = original_run_game(*args, **kwargs)
                if not advanced["value"]:
                    advanced["value"] = True
                    train(
                        env_bin=launcher,
                        out_dir=store,
                        resume=store / "latest.json",
                        until_update=1,
                    )
                return row

            with mock.patch.object(evaluator_mod, "_run_game", side_effect=advancing_run_game):
                result = _evaluate_fixture(root, launcher, store, head)
            self.assertTrue(advanced["value"])
            self.assertTrue((result / "run.json").is_file())
            self.assertEqual(TrainingStore(store).validate_latest().head.update, 1)
            manifest = read_json_file(result / "run.json")
            self.assertEqual([snapshot["update"] for snapshot in manifest["snapshots"]], [0, 0])
            self.assertEqual(manifest["snapshots"][0]["head"], head)
            validate_evaluation(result)

    def test_selected_generation_replacement_fails_before_artifact_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, source_store, head = _train_fixture(root, scenario="train_pair", until_update=1)
            for role in ("candidate", "baseline"):
                for path_attr in ("update_path", "sidecar_path", "checkpoint_path"):
                    with self.subTest(role=role, artifact=path_attr):
                        case = f"mutation-{role}-{path_attr}"
                        store = root / f"store-{case}"
                        shutil.copytree(source_store, store)
                        chain = TrainingStore(store).validate_latest()
                        selected = chain.head if role == "candidate" else chain.snapshots[0]
                        target = getattr(selected, path_attr)
                        original_run_game = evaluator_mod._run_game
                        calls = {"count": 0}

                        def replacing_run_game(*args, **kwargs):  # type: ignore[no-untyped-def]
                            row = original_run_game(*args, **kwargs)
                            calls["count"] += 1
                            if calls["count"] == 2:
                                replacement = target.with_name(f"replacement-{target.name}")
                                replacement.write_bytes(target.read_bytes() + b"tampered")
                                os.replace(replacement, target)
                            return row

                        out = root / case
                        with mock.patch.object(evaluator_mod, "_run_game", side_effect=replacing_run_game):
                            with self.assertRaises((ValueError, RuntimeError)):
                                _evaluate_fixture(root, launcher, store, head, out_name=case)
                        self.assertEqual(calls["count"], 2)
                        self.assertEqual({path.name for path in out.iterdir()}, {OUTPUT_LOCK_FILE_NAME})

    def test_environment_mutation_fails_before_artifact_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            original_bytes = launcher.read_bytes()
            original_run_game = evaluator_mod._run_game
            calls = {"count": 0}

            def mutating_run_game(*args, **kwargs):  # type: ignore[no-untyped-def]
                row = original_run_game(*args, **kwargs)
                calls["count"] += 1
                if calls["count"] == 2:
                    launcher.write_bytes(original_bytes + b"\n# mutation\n")
                return row

            out = root / "evaluation"
            try:
                with mock.patch.object(evaluator_mod, "_run_game", side_effect=mutating_run_game):
                    with self.assertRaisesRegex(ValueError, "environment binary changed"):
                        _evaluate_fixture(root, launcher, store, head)
            finally:
                launcher.write_bytes(original_bytes)
            self.assertEqual(calls["count"], 2)
            self.assertEqual({path.name for path in out.iterdir()}, {OUTPUT_LOCK_FILE_NAME})

    def test_protocol_and_process_failures_leave_lock_only_and_are_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            scenario_file = root / "scenario.txt"
            scenario_file.write_text("valid", encoding="utf-8")
            launcher, store, head = _train_fixture(
                root,
                scenario="switchable",
                extra_env={"FAKE_SCENARIO_FILE": str(scenario_file)},
            )
            for scenario in (
                "duplicate_keys",
                "noise",
                "nonfinite_json",
                "nonfinite_overflow",
                "truncated_terminal",
                "halted_terminal",
                "eof_nonzero",
                "timeout",
                "provenance_drift",
            ):
                with self.subTest(scenario=scenario):
                    out_name = f"failure-{scenario}"
                    out = root / out_name
                    scenario_file.write_text(scenario, encoding="utf-8")
                    timeout_ms = 100 if scenario == "timeout" else 5_000
                    with self.assertRaises(Exception):
                        evaluate(
                            training_store=store,
                            expected_candidate_head=head,
                            env_bin=launcher,
                            out_dir=out,
                            pairs=1,
                            base_seed=4,
                            bootstrap_replicates=1_000,
                            max_decisions=8,
                            timeout_ms=timeout_ms,
                        )
                    self.assertEqual({path.name for path in out.iterdir()}, {OUTPUT_LOCK_FILE_NAME})
                    scenario_file.write_text("valid", encoding="utf-8")
                    _evaluate_fixture(root, launcher, store, head, out_name=out_name)
                    validated = validate_evaluation(out)
                    self.assertEqual((validated.total_half_points, validated.estimate), (2, 0.5))


class EvaluatorUnitTest(unittest.TestCase):
    def _decision(self) -> Decision:
        return Decision(0, 0, "p0", actor_observation("p0"), legal_actions("p0"), dict(PROVENANCE))

    def test_greedy_tie_nonfinite_shape_and_global_rng(self) -> None:
        decision = self._decision()

        class FixedModel:
            def __init__(self, logits: torch.Tensor, value: torch.Tensor) -> None:
                self.logits = logits
                self.value = value

            def __call__(self, _encoded):  # type: ignore[no-untyped-def]
                return self.logits, self.value

        random.seed(12)
        torch.manual_seed(34)
        python_rng = random.getstate()
        torch_rng = torch.get_rng_state().clone()
        self.assertEqual(
            _select_greedy_action(FixedModel(torch.tensor([2.0, 2.0, 1.0]), torch.tensor(0.0)), decision),
            0,
        )
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        bad = (
            FixedModel(torch.tensor([0.0, float("nan"), 1.0]), torch.tensor(0.0)),
            FixedModel(torch.tensor([0.0, 1.0]), torch.tensor(0.0)),
            FixedModel(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64), torch.tensor(0.0)),
            FixedModel(torch.tensor([0.0, 1.0, 2.0]), torch.tensor(float("inf"))),
            FixedModel(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.0])),
        )
        for model in bad:
            with self.subTest(model=model), self.assertRaises(ValueError):
                _select_greedy_action(model, decision)

    def test_exact_request_bounds_reject_bool_and_product_overflow(self) -> None:
        valid = dict(
            expected_candidate_head="a" * 64,
            pairs=1,
            base_seed=0,
            bootstrap_replicates=1_000,
            max_decisions=1,
            timeout_ms=1,
        )
        for key, value in (
            ("pairs", True),
            ("base_seed", True),
            ("bootstrap_replicates", True),
            ("max_decisions", True),
            ("timeout_ms", True),
            ("timeout_ms", 0),
            ("pairs", 50_001),
            ("bootstrap_replicates", 999),
            ("max_decisions", 10_000_001),
        ):
            args = {**valid, key: value}
            with self.subTest(key=key, value=value), self.assertRaises((TypeError, ValueError)):
                _validate_request(**args)
        with self.assertRaises(ValueError):
            _validate_request(**{**valid, "pairs": 501, "bootstrap_replicates": 100_000})
        for head in ("A" * 64, "a" * 63, 1):
            with self.subTest(head=head), self.assertRaises(ValueError):
                _validate_request(**{**valid, "expected_candidate_head": head})

    def test_cli_exact_surface_and_path_free_canonical_summary(self) -> None:
        argv = [
            "evaluate",
            "--training-store",
            "source",
            "--expected-candidate-head",
            "a" * 64,
            "--env-bin",
            "env",
            "--out-dir",
            "output",
            "--pairs",
            "2",
            "--base-seed",
            "3",
            "--bootstrap-replicates",
            "1000",
            "--max-decisions",
            "8",
            "--timeout-ms",
            "5000",
        ]
        parsed = cli_mod.build_parser().parse_args(argv)
        self.assertEqual(parsed.command, "evaluate")
        for forbidden in ("baseline", "mode", "resume"):
            self.assertFalse(hasattr(parsed, forbidden))
        result = EvaluationResult("b" * 64, "a" * 64, "c" * 64, 2, 4, 5, 0.625)
        output = io.StringIO()
        with mock.patch.object(cli_mod, "evaluate", return_value=result) as called, contextlib.redirect_stdout(output):
            self.assertEqual(cli_mod.main(argv), 0)
        called.assert_called_once()
        line = output.getvalue()
        self.assertEqual(line, json.dumps(json.loads(line), sort_keys=True, separators=(",", ":")) + "\n")
        self.assertNotIn("source", line)
        self.assertNotIn("output", line)
        self.assertEqual(json.loads(line)["estimate_hex"], (0.625).hex())


class EvaluationVerifierCorruptionTest(unittest.TestCase):
    def test_verifier_rejects_canonical_hash_order_stats_pair_and_root_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            pristine = root / "pristine"
            evaluate(
                training_store=store,
                expected_candidate_head=head,
                env_bin=launcher,
                out_dir=pristine,
                pairs=1,
                base_seed=4,
                bootstrap_replicates=1_000,
                max_decisions=8,
                timeout_ms=5_000,
            )
            validate_evaluation(pristine)

            def clone(name: str) -> Path:
                target = root / name
                shutil.copytree(pristine, target)
                return target

            target = clone("extra")
            (target / "extra.json").write_text("{}\n", encoding="ascii")
            with self.assertRaises(ValueError):
                validate_evaluation(target)

            target = clone("missing-run")
            (target / "run.json").unlink()
            with self.assertRaises(ValueError):
                validate_evaluation(target)

            target = clone("noncanonical")
            (target / "run.json").write_bytes((target / "run.json").read_bytes().rstrip(b"\n") + b" \n")
            with self.assertRaises(ValueError):
                validate_evaluation(target)

            target = clone("duplicate-key")
            data = (target / "games.jsonl").read_bytes()
            (target / "games.jsonl").write_bytes(data.replace(b"{", b'{"schema":"duplicate",', 1))
            with self.assertRaises(ValueError):
                validate_evaluation(target)

            target = clone("order")
            lines = (target / "games.jsonl").read_bytes().splitlines(keepends=True)
            (target / "games.jsonl").write_bytes(lines[1] + lines[0])
            _rewrite_manifest_file_metadata(target, "games.jsonl")
            with self.assertRaises(ValueError):
                validate_evaluation(target)

            target = clone("hash")
            manifest = read_json_file(target / "run.json")
            manifest["files"]["games.jsonl"]["sha256"] = "0" * 64
            (target / "run.json").write_bytes(canonical_json_bytes(manifest))
            with self.assertRaises(ValueError):
                validate_evaluation(target)

            target = clone("statistics")
            manifest = read_json_file(target / "run.json")
            manifest["statistics"]["paired"]["total_half_points"] += 1
            (target / "run.json").write_bytes(canonical_json_bytes(manifest))
            with self.assertRaises(ValueError):
                validate_evaluation(target)

            target = clone("pair")
            pair = _jsonl(target / "pairs.jsonl")[0]
            pair["total_half_points"] = 3
            (target / "pairs.jsonl").write_bytes(canonical_json_bytes(pair))
            _rewrite_manifest_file_metadata(target, "pairs.jsonl")
            with self.assertRaises(ValueError):
                validate_evaluation(target)

            for name, mutate in (
                ("bool-game", lambda manifest, games, pairs: games[0]["terminal_reward"].__setitem__(0, True)),
                ("float-game", lambda manifest, games, pairs: games[0]["terminal_reward"].__setitem__(0, 1.0)),
                ("bool-pair", lambda manifest, games, pairs: pairs[0].__setitem__("candidate_as_p1_half_points", False)),
                ("bool-stat", lambda manifest, games, pairs: manifest["statistics"]["games"].__setitem__("draws", False)),
                ("bool-scoring", lambda manifest, games, pairs: manifest["scoring"].__setitem__("candidate_draw", True)),
                ("int-publication", lambda manifest, games, pairs: manifest["publication"].__setitem__("resume", 0)),
                (
                    "bool-runtime",
                    lambda manifest, games, pairs: (
                        manifest["source_training"]["runtime_compatibility"].__setitem__("num_threads", True),
                        manifest["evaluator_runtime_compatibility"].__setitem__("num_threads", True),
                    ),
                ),
            ):
                target = clone(name)
                manifest = read_json_file(target / "run.json")
                games = _jsonl(target / "games.jsonl")
                pairs = _jsonl(target / "pairs.jsonl")
                mutate(manifest, games, pairs)
                (target / "games.jsonl").write_bytes(b"".join(canonical_json_bytes(row) for row in games))
                (target / "pairs.jsonl").write_bytes(b"".join(canonical_json_bytes(row) for row in pairs))
                for file_name, rows in (("games.jsonl", games), ("pairs.jsonl", pairs)):
                    data = (target / file_name).read_bytes()
                    manifest["files"][file_name] = {
                        "row_count": len(rows),
                        "sha256": sha256_bytes(data),
                        "size_bytes": len(data),
                    }
                (target / "run.json").write_bytes(canonical_json_bytes(manifest))
                with self.subTest(type_drift=name), self.assertRaises(ValueError):
                    validate_evaluation(target)


class EvaluationFilesystemBoundaryTest(unittest.TestCase):
    def test_committed_verifier_rejects_file_symlinks_and_preserves_targets(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("file symlink primitive unavailable")
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            outside = root / "outside"
            outside.mkdir()
            probe_target = outside / "probe-target"
            probe_target.write_bytes(b"probe")
            probe = root / "probe-link"
            try:
                os.symlink(probe_target, probe)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlink unavailable: {exc}")
            else:
                probe.unlink()
            launcher, store, head = _train_fixture(root)
            pristine = _evaluate_fixture(root, launcher, store, head, out_name="pristine-symlink")
            for name in (OUTPUT_LOCK_FILE_NAME, "games.jsonl", "pairs.jsonl", "run.json"):
                with self.subTest(name=name):
                    target = root / f"symlink-{name.replace('.', '-')}"
                    shutil.copytree(pristine, target)
                    artifact = target / name
                    sentinel = outside / f"{name.replace('.', '-')}.sentinel"
                    sentinel.write_bytes(artifact.read_bytes())
                    expected = sentinel.read_bytes()
                    artifact.unlink()
                    os.symlink(sentinel, artifact)
                    with self.assertRaises(ValueError):
                        validate_evaluation(target)
                    self.assertEqual(sentinel.read_bytes(), expected)

    def test_committed_verifier_rejects_hardlinks_and_preserves_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            outside = root / "outside"
            outside.mkdir()
            launcher, store, head = _train_fixture(root)
            pristine = _evaluate_fixture(root, launcher, store, head, out_name="pristine-hardlink")
            for name in (OUTPUT_LOCK_FILE_NAME, "games.jsonl", "pairs.jsonl", "run.json"):
                with self.subTest(name=name):
                    target = root / f"hardlink-{name.replace('.', '-')}"
                    shutil.copytree(pristine, target)
                    artifact = target / name
                    sentinel = outside / f"{name.replace('.', '-')}.hardlink"
                    sentinel.write_bytes(artifact.read_bytes())
                    expected = sentinel.read_bytes()
                    artifact.unlink()
                    try:
                        os.link(sentinel, artifact)
                    except OSError as exc:
                        self.skipTest(f"hardlink unavailable on temporary filesystem: {exc}")
                    with self.assertRaises(ValueError):
                        validate_evaluation(target)
                    self.assertEqual(sentinel.read_bytes(), expected)

    def test_root_junction_or_symlink_is_rejected_for_verification_and_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            pristine = _evaluate_fixture(root, launcher, store, head, out_name="pristine-root-alias")
            before = _artifact_bytes(pristine)
            verify_alias = root / "verify-root-alias"
            _create_directory_alias(self, verify_alias, pristine)
            try:
                with self.assertRaises(ValueError):
                    validate_evaluation(verify_alias)
                self.assertEqual(_artifact_bytes(pristine), before)
            finally:
                _remove_directory_alias(verify_alias)

            outside = root / "outside-publication"
            outside.mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_bytes(b"external-sentinel")
            publish_alias = root / "publish-root-alias"
            _create_directory_alias(self, publish_alias, outside)
            try:
                with self.assertRaises(ValueError):
                    _evaluate_fixture(root, launcher, store, head, out_name=publish_alias.name)
                self.assertEqual(sentinel.read_bytes(), b"external-sentinel")
                self.assertEqual({path.name for path in outside.iterdir()}, {"sentinel.txt"})
            finally:
                _remove_directory_alias(publish_alias)

    def test_ancestor_directory_alias_with_real_final_component_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            real_parent = root / "real-parent"
            real_parent.mkdir()
            launcher, store, head = _train_fixture(real_parent)
            alias_parent = root / "alias-parent"
            _create_directory_alias(self, alias_parent, real_parent)
            try:
                result = _evaluate_fixture(alias_parent, launcher, store, head)
                self.assertEqual(result, alias_parent / "evaluation")
                validated = validate_evaluation(real_parent / "evaluation")
                self.assertEqual((validated.total_half_points, validated.estimate), (2, 0.5))
            finally:
                _remove_directory_alias(alias_parent)

    def test_exact_root_types_and_preexisting_aliases_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            pristine = _evaluate_fixture(root, launcher, store, head, out_name="pristine-exact-root")

            wrong_type = root / "wrong-type"
            shutil.copytree(pristine, wrong_type)
            (wrong_type / "games.jsonl").unlink()
            (wrong_type / "games.jsonl").mkdir()
            with self.assertRaises(ValueError):
                validate_evaluation(wrong_type)

            outside = root / "outside"
            outside.mkdir()
            sentinel = outside / "sentinel.bin"
            sentinel.write_bytes(b"do-not-change")
            hardlink_root = root / "publish-hardlink"
            hardlink_root.mkdir()
            try:
                os.link(sentinel, hardlink_root / "run.json")
            except OSError as exc:
                self.skipTest(f"hardlink unavailable on temporary filesystem: {exc}")
            with self.assertRaises(FileExistsError):
                _evaluate_fixture(root, launcher, store, head, out_name=hardlink_root.name)
            self.assertEqual(sentinel.read_bytes(), b"do-not-change")

            directory_root = root / "publish-directory"
            (directory_root / "run.json").mkdir(parents=True)
            nested_sentinel = directory_root / "run.json" / "sentinel.txt"
            nested_sentinel.write_bytes(b"nested-sentinel")
            with self.assertRaises(FileExistsError):
                _evaluate_fixture(root, launcher, store, head, out_name=directory_root.name)
            self.assertEqual(nested_sentinel.read_bytes(), b"nested-sentinel")

    def test_preexisting_publication_file_symlink_is_not_followed(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("file symlink primitive unavailable")
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            outside = root / "outside-symlink"
            outside.mkdir()
            sentinel = outside / "sentinel.json"
            sentinel.write_bytes(b"external-sentinel")
            out = root / "publication-symlink"
            out.mkdir()
            try:
                os.symlink(sentinel, out / "run.json")
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlink unavailable: {exc}")
            launcher, store, head = _train_fixture(root)
            with self.assertRaises((FileExistsError, ValueError)):
                _evaluate_fixture(root, launcher, store, head, out_name=out.name)
            self.assertEqual(sentinel.read_bytes(), b"external-sentinel")

    def test_child_junction_or_directory_symlink_is_rejected_without_following(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            pristine = _evaluate_fixture(root, launcher, store, head, out_name="pristine-child-alias")
            outside = root / "outside-child"
            outside.mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_bytes(b"external-child-sentinel")
            target = root / "child-alias"
            shutil.copytree(pristine, target)
            (target / "games.jsonl").unlink()
            alias = target / "games.jsonl"
            _create_directory_alias(self, alias, outside)
            try:
                with self.assertRaises(ValueError):
                    validate_evaluation(target)
                self.assertEqual(sentinel.read_bytes(), b"external-child-sentinel")
            finally:
                _remove_directory_alias(alias)


class EvaluationRealEnvironmentTest(unittest.TestCase):
    def test_real_environment_update_zero_aa_is_neutral_and_source_unchanged(self) -> None:
        env_value = os.environ.get("MTG_KERNEL_RL_ENV_BIN")
        if not env_value:
            self.skipTest("MTG_KERNEL_RL_ENV_BIN not set")
        env_bin = Path(env_value)
        self.assertTrue(env_bin.is_file(), f"MTG_KERNEL_RL_ENV_BIN is not a file: {env_bin}")
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            store = root / "training-store"
            train(
                env_bin=env_bin,
                out_dir=store,
                base_seed=71_501,
                until_update=0,
                batch_episodes=2,
                learning_rate=0.001,
                value_coef=0.5,
                max_decisions=5_000,
            )
            chain = TrainingStore(store).validate_latest()
            self.assertEqual(chain.head.update, 0)
            before = _tree_bytes(store)
            result = evaluate(
                training_store=store,
                expected_candidate_head=chain.head.head,
                env_bin=env_bin,
                out_dir=root / "evaluation",
                pairs=1,
                base_seed=71_501,
                bootstrap_replicates=1_000,
                max_decisions=5_000,
                timeout_ms=60_000,
            )
            self.assertEqual(_tree_bytes(store), before)
            self.assertEqual(result.candidate_head, result.baseline_head)
            self.assertEqual((result.pair_count, result.game_count, result.total_half_points), (1, 2, 2))
            self.assertEqual(result.estimate.hex(), (0.5).hex())
            self.assertEqual(validate_evaluation(root / "evaluation").run_sha256, result.run_sha256)


if __name__ == "__main__":
    unittest.main()
