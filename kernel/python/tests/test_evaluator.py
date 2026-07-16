from __future__ import annotations

import contextlib
import hashlib
import io
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import mtg_kernel_rl.cli as cli_mod
import mtg_kernel_rl.evaluator as evaluator_mod
from mtg_kernel_rl.artifact_io import canonical_json_bytes, read_json_file, sha256_bytes, validate_training_json_privacy
from mtg_kernel_rl.client import Decision
from mtg_kernel_rl.determinism import derive_evaluation_env_seed
from mtg_kernel_rl.evaluation_store import validate_evaluation
from mtg_kernel_rl.evaluator import EvaluationResult, _select_greedy_action, _validate_request, evaluate
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


if __name__ == "__main__":
    unittest.main()
