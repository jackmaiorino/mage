from __future__ import annotations

import contextlib
import io
import json
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import mtg_kernel_rl.cli as cli_mod
import mtg_kernel_rl.sampled_evaluator as sampled_mod
from mtg_kernel_rl.artifact_io import canonical_json_bytes, read_json_file, sha256_bytes
from mtg_kernel_rl.client import Decision, Terminal
from mtg_kernel_rl.determinism import derive_evaluation_action_seed, derive_evaluation_bootstrap_seed
from mtg_kernel_rl.evaluation_stats import PairedGamePoints, score_pair_half_points, summarize_paired_game_points
from mtg_kernel_rl.evaluation_store import statistics_payload, validate_evaluation
from mtg_kernel_rl.evaluator import EvaluationResult, evaluate
from mtg_kernel_rl.sampled_evaluation_store import (
    ACTION_SEED_DERIVATION_CONTRACT,
    ACTION_SELECTION_CONTRACT,
    ALGORITHM_CONTRACT,
    GAME_SCHEMA,
    PAIR_SCHEMA,
    RUN_SCHEMA,
    SEAT_SCHEDULE_CONTRACT,
    validate_sampled_evaluation,
)
from mtg_kernel_rl.sampled_evaluator import _run_sampled_game, _select_sampled_action, evaluate_sampled
from mtg_kernel_rl.trainer import train
from mtg_kernel_rl.training_store import TrainingStore

from fixtures import PROVENANCE, actor_observation, fake_launcher, legal_actions


def _train_fixture(
    root: Path,
    *,
    scenario: str = "valid",
    until_update: int = 0,
    max_decisions: int = 8,
) -> tuple[Path, Path, str]:
    launcher = fake_launcher(root, scenario)
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


def _evaluate_fixture(
    launcher: Path,
    store: Path,
    head: str,
    out: Path,
    *,
    pairs: int = 2,
    base_seed: int = 4,
    max_decisions: int = 8,
    timeout_ms: int = 5_000,
) -> EvaluationResult:
    return evaluate_sampled(
        training_store=store,
        expected_candidate_head=head,
        env_bin=launcher,
        out_dir=out,
        pairs=pairs,
        base_seed=base_seed,
        bootstrap_replicates=1_000,
        max_decisions=max_decisions,
        timeout_ms=timeout_ms,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def _refresh_file_metadata(root: Path, manifest: dict, name: str) -> None:
    data = (root / name).read_bytes()
    manifest["files"][name] = {
        "row_count": len(data.splitlines()),
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }


class SampledSelectorTest(unittest.TestCase):
    def _decision(self) -> Decision:
        return Decision(0, 0, "p0", actor_observation("p0"), legal_actions("p0"), dict(PROVENANCE))

    def test_selector_goldens_repeatability_seed_separation_and_global_rng(self) -> None:
        decision = self._decision()

        class FixedModel:
            def __call__(self, _encoded):  # type: ignore[no-untyped-def]
                return torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32), torch.tensor(0.0)

        seeds = (
            derive_evaluation_action_seed(71_501, 0, "p0", 0),
            derive_evaluation_action_seed(71_501, 0, "p1", 0),
            derive_evaluation_action_seed(71_501, 0, "p0", 1),
            derive_evaluation_action_seed(71_501, 1, "p0", 0),
            derive_evaluation_action_seed(0, 0, "p0", 0),
            derive_evaluation_action_seed((1 << 63) - 1, (1 << 63) - 1, "p1", (1 << 63) - 1),
        )
        random.seed(81)
        torch.manual_seed(82)
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        selected = [_select_sampled_action(FixedModel(), decision, seed) for seed in seeds]
        self.assertEqual(selected, [1, 1, 2, 2, 2, 0])
        self.assertEqual(_select_sampled_action(FixedModel(), decision, seeds[0]), selected[0])
        self.assertGreater(len(set(selected)), 1)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_selector_rejects_bad_logits_values_and_seed_types(self) -> None:
        decision = self._decision()

        class FixedModel:
            def __init__(self, output):  # type: ignore[no-untyped-def]
                self.output = output

            def __call__(self, _encoded):  # type: ignore[no-untyped-def]
                return self.output

        bad_outputs = (
            [0.0, 1.0, 2.0],
            (torch.tensor([0.0, 1.0, 2.0]),),
            (torch.tensor([0.0, 1.0]), torch.tensor(0.0)),
            (torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64), torch.tensor(0.0)),
            (torch.tensor([0.0, float("nan"), 2.0]), torch.tensor(0.0)),
            (torch.tensor([0.0, float("inf"), 2.0]), torch.tensor(0.0)),
            (torch.tensor([0.0, 1.0, 2.0]), 0.0),
            (torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.0])),
            (torch.tensor([0.0, 1.0, 2.0]), torch.tensor(0.0, dtype=torch.float64)),
            (torch.tensor([0.0, 1.0, 2.0]), torch.tensor(float("nan"))),
        )
        for output in bad_outputs:
            with self.subTest(output=output), self.assertRaises((TypeError, ValueError)):
                _select_sampled_action(FixedModel(output), decision, 0)
        for bad_seed in (True, -1, 2**63):
            with self.subTest(action_seed=bad_seed), self.assertRaises((TypeError, ValueError)):
                _select_sampled_action(
                    FixedModel((torch.tensor([0.0, 1.0, 2.0]), torch.tensor(0.0))),
                    decision,
                    bad_seed,  # type: ignore[arg-type]
                )
        valid_model = FixedModel((torch.tensor([0.0, 1.0, 2.0]), torch.tensor(0.0)))
        with mock.patch.object(torch, "multinomial", return_value=torch.tensor([0], dtype=torch.int32)):
            with self.assertRaisesRegex(ValueError, "invalid selection"):
                _select_sampled_action(valid_model, decision, 0)


class SampledCrnRoutingTest(unittest.TestCase):
    def test_role_swap_preserves_physical_action_seed_streams(self) -> None:
        actors = ("p0", "p1", "p0")

        class ScriptedClient:
            def __init__(self) -> None:
                self.index = 0
                self.episode_id = 0

            def _decision(self) -> Decision:
                actor = actors[self.index]
                return Decision(
                    self.episode_id,
                    self.index,
                    actor,
                    actor_observation(actor, self.index),
                    legal_actions(actor),
                    dict(PROVENANCE),
                )

            def reset(self, *, episode_id: int, env_seed: int, max_decisions: int):  # type: ignore[no-untyped-def]
                self.index = 0
                self.episode_id = episode_id
                return self._decision()

            def step(self, _selected_index: int, _stable_id: str):  # type: ignore[no-untyped-def]
                self.index += 1
                if self.index < len(actors):
                    return self._decision()
                return Terminal(
                    self.episode_id,
                    "p0_win",
                    "natural",
                    "natural_game_over",
                    "p0",
                    [1, -1],
                    len(actors),
                    dict(PROVENANCE),
                )

        candidate = object()
        baseline = object()
        events: list[tuple[int, str, object, int]] = []
        current_leg = {"value": -1}

        def select(model, decision, seed):  # type: ignore[no-untyped-def]
            events.append((current_leg["value"], decision.acting_player, model, seed))
            return 0

        with mock.patch.object(sampled_mod, "_select_sampled_action", side_effect=select):
            current_leg["value"] = 0
            first = _run_sampled_game(
                ScriptedClient(),
                pair_index=7,
                game_in_pair=0,
                env_seed=123,
                base_seed=71_501,
                max_decisions=8,
                candidate_model=candidate,  # type: ignore[arg-type]
                baseline_model=baseline,  # type: ignore[arg-type]
                expected_provenance=PROVENANCE,
            )
            current_leg["value"] = 1
            second = _run_sampled_game(
                ScriptedClient(),
                pair_index=7,
                game_in_pair=1,
                env_seed=123,
                base_seed=71_501,
                max_decisions=8,
                candidate_model=candidate,  # type: ignore[arg-type]
                baseline_model=baseline,  # type: ignore[arg-type]
                expected_provenance=PROVENANCE,
            )

        first_events = events[: len(actors)]
        second_events = events[len(actors) :]
        self.assertEqual(
            [(seat, seed) for _leg, seat, _model, seed in first_events],
            [(seat, seed) for _leg, seat, _model, seed in second_events],
        )
        expected_stream = [
            ("p0", derive_evaluation_action_seed(71_501, 7, "p0", 0)),
            ("p1", derive_evaluation_action_seed(71_501, 7, "p1", 0)),
            ("p0", derive_evaluation_action_seed(71_501, 7, "p0", 1)),
        ]
        self.assertEqual([(seat, seed) for _leg, seat, _model, seed in first_events], expected_stream)
        self.assertEqual([model for _leg, _seat, model, _seed in first_events], [candidate, baseline, candidate])
        self.assertEqual([model for _leg, _seat, model, _seed in second_events], [baseline, candidate, baseline])
        self.assertEqual(first["candidate_decisions"], second["baseline_decisions"])
        self.assertEqual(first["baseline_decisions"], second["candidate_decisions"])


class SampledEndToEndTest(unittest.TestCase):
    def test_aa_is_neutral_deterministic_strict_and_rng_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            before_store = _tree_bytes(store)
            random.seed(90)
            torch.manual_seed(91)
            python_state = random.getstate()
            torch_state = torch.get_rng_state().clone()
            first_result = _evaluate_fixture(launcher, store, head, root / "sampled-a")
            self.assertEqual(random.getstate(), python_state)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            second_result = _evaluate_fixture(launcher, store, head, root / "sampled-b")
            self.assertEqual(_tree_bytes(root / "sampled-a"), _tree_bytes(root / "sampled-b"))
            self.assertEqual(first_result, second_result)
            self.assertEqual(_tree_bytes(store), before_store)

            validated = validate_sampled_evaluation(root / "sampled-a")
            self.assertEqual((validated.pair_count, validated.game_count), (2, 4))
            self.assertEqual((validated.total_half_points, validated.estimate), (4, 0.5))
            manifest = read_json_file(root / "sampled-a" / "run.json")
            self.assertEqual(manifest["schema"], RUN_SCHEMA)
            self.assertEqual(manifest["algorithm"], ALGORITHM_CONTRACT)
            self.assertEqual(manifest["artifact_schemas"], {"game": GAME_SCHEMA, "pair": PAIR_SCHEMA, "run": RUN_SCHEMA})
            self.assertEqual(manifest["action_seed_derivation"], ACTION_SEED_DERIVATION_CONTRACT)
            self.assertEqual(manifest["action_selection"], ACTION_SELECTION_CONTRACT)
            self.assertEqual(manifest["seat_schedule"], SEAT_SCHEDULE_CONTRACT)
            games = _jsonl(root / "sampled-a" / "games.jsonl")
            pairs = _jsonl(root / "sampled-a" / "pairs.jsonl")
            self.assertTrue(all(len(row) == 17 and row["schema"] == GAME_SCHEMA for row in games))
            self.assertTrue(all(len(row) == 8 and row["schema"] == PAIR_SCHEMA for row in pairs))
            for pair_index, pair in enumerate(pairs):
                first, second = games[2 * pair_index : 2 * pair_index + 2]
                self.assertEqual(pair["total_half_points"], 2)
                self.assertEqual(first["decision_count"], second["decision_count"])
                self.assertEqual(first["candidate_decisions"], second["baseline_decisions"])
                self.assertEqual(first["baseline_decisions"], second["candidate_decisions"])
                for key in ("terminal_outcome", "terminal_classification", "terminal_code", "terminal_reward", "winner"):
                    self.assertEqual(first[key], second[key])
            with self.assertRaises(ValueError):
                validate_evaluation(root / "sampled-a")

    def test_v1_artifact_tree_is_byte_unchanged_by_sampled_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            v1_root = root / "v1-evaluation"
            evaluate(
                training_store=store,
                expected_candidate_head=head,
                env_bin=launcher,
                out_dir=v1_root,
                pairs=1,
                base_seed=4,
                bootstrap_replicates=1_000,
                max_decisions=8,
                timeout_ms=5_000,
            )
            before = _tree_bytes(v1_root)
            _evaluate_fixture(launcher, store, head, root / "sampled", pairs=1)
            validate_sampled_evaluation(root / "sampled")
            validate_evaluation(v1_root)
            self.assertEqual(_tree_bytes(v1_root), before)

    def test_distinct_head_ab_evaluation_accepts_nonidentical_legs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root, scenario="train_pair", until_update=1)
            chain = TrainingStore(store).validate_latest()
            self.assertNotEqual(chain.head.head, chain.snapshots[0].head)
            before_store = _tree_bytes(store)
            result = _evaluate_fixture(launcher, store, head, root / "sampled-ab", pairs=1)
            validated = validate_sampled_evaluation(root / "sampled-ab")
            self.assertEqual(result.candidate_head, chain.head.head)
            self.assertEqual(result.baseline_head, chain.snapshots[0].head)
            self.assertEqual(validated.run_sha256, result.run_sha256)
            games = _jsonl(root / "sampled-ab" / "games.jsonl")
            self.assertNotEqual(games[0]["terminal_outcome"], games[1]["terminal_outcome"])
            self.assertEqual(_tree_bytes(store), before_store)

    def test_cli_surface_and_canonical_summary(self) -> None:
        argv = [
            "evaluate-sampled",
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
        self.assertEqual(parsed.command, "evaluate-sampled")
        for forbidden in ("baseline", "mode", "temperature", "resume"):
            self.assertFalse(hasattr(parsed, forbidden))
        result = EvaluationResult("b" * 64, "a" * 64, "c" * 64, 2, 4, 5, 0.625)
        output = io.StringIO()
        with mock.patch.object(cli_mod, "evaluate_sampled", return_value=result) as called, contextlib.redirect_stdout(output):
            self.assertEqual(cli_mod.main(argv), 0)
        called.assert_called_once()
        line = output.getvalue()
        self.assertEqual(line, json.dumps(json.loads(line), sort_keys=True, separators=(",", ":")) + "\n")
        self.assertEqual(
            set(json.loads(line)),
            {"baseline_head", "candidate_head", "estimate_hex", "game_count", "pair_count", "run_sha256", "total_half_points"},
        )


class SampledVerifierCorruptionTest(unittest.TestCase):
    def test_contract_manifest_and_row_corruption_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            pristine = root / "pristine"
            _evaluate_fixture(launcher, store, head, pristine, pairs=1)

            def clone(name: str) -> Path:
                target = root / name
                shutil.copytree(pristine, target)
                return target

            manifest_mutations = (
                ("run-schema", lambda value: value.__setitem__("schema", "kernel_rl_paired_evaluation/v1")),
                ("artifact-game", lambda value: value["artifact_schemas"].__setitem__("game", "bad")),
                ("artifact-pair", lambda value: value["artifact_schemas"].__setitem__("pair", "bad")),
                ("artifact-run", lambda value: value["artifact_schemas"].__setitem__("run", "bad")),
                ("algorithm-name", lambda value: value["algorithm"].__setitem__("name", "bad")),
                ("algorithm-primary", lambda value: value["algorithm"].__setitem__("primary_statistic", "bad")),
                ("algorithm-intervals", lambda value: value["algorithm"].__setitem__("descriptive_intervals", "bad")),
                ("action-seed-version", lambda value: value["action_seed_derivation"].__setitem__("version", "bad")),
                ("action-seed-algorithm", lambda value: value["action_seed_derivation"].__setitem__("algorithm", "bad")),
                ("action-seed-namespace", lambda value: value["action_seed_derivation"]["namespaces"].__setitem__(0, "bad")),
                ("action-seed-seat", lambda value: value["action_seed_derivation"]["physical_seat_encoding"].__setitem__("p0", 1)),
                ("selection-mode", lambda value: value["action_selection"].__setitem__("mode", "greedy")),
                ("selection-inference", lambda value: value["action_selection"].__setitem__("inference", "bad")),
                ("selection-temperature", lambda value: value["action_selection"].__setitem__("temperature_hex", "0x0.0p+0")),
                ("selection-replacement", lambda value: value["action_selection"].__setitem__("replacement", True)),
                ("selection-rng", lambda value: value["action_selection"].__setitem__("action_rng", "bad")),
                ("selection-algorithm", lambda value: value["action_selection"].__setitem__("algorithm", "bad")),
                ("seat-candidate-p0", lambda value: value["seat_schedule"].__setitem__("candidate_as_p0", "bad")),
                ("seat-candidate-p1", lambda value: value["seat_schedule"].__setitem__("candidate_as_p1", "bad")),
                ("seat-env", lambda value: value["seat_schedule"].__setitem__("paired_environment_seed", "bad")),
                ("seat-action", lambda value: value["seat_schedule"].__setitem__("paired_physical_action_streams", "bad")),
                ("environment", lambda value: value["environment"].__setitem__("binary_sha256", "0" * 64)),
                ("statistics", lambda value: value["statistics"]["paired"].__setitem__("total_half_points", 0)),
                ("files", lambda value: value["files"]["games.jsonl"].__setitem__("sha256", "0" * 64)),
                ("bool-type", lambda value: value["configuration"].__setitem__("base_seed", True)),
                ("privacy", lambda value: value["action_selection"].__setitem__("mode", "file:///external/secret")),
            )
            for name, mutate in manifest_mutations:
                target = clone(f"manifest-{name}")
                manifest = read_json_file(target / "run.json")
                mutate(manifest)
                (target / "run.json").write_bytes(canonical_json_bytes(manifest))
                with self.subTest(manifest=name), self.assertRaises((TypeError, ValueError)):
                    validate_sampled_evaluation(target)

            game_mutations = (
                ("schema", lambda row: row.__setitem__("schema", "bad")),
                ("pair", lambda row: row.__setitem__("pair_index", 1)),
                ("episode", lambda row: row.__setitem__("episode_id", 2)),
                ("env", lambda row: row.__setitem__("env_seed", row["env_seed"] + 1)),
                ("candidate-seat", lambda row: row.__setitem__("candidate_seat", "p1")),
                ("baseline-seat", lambda row: row.__setitem__("baseline_seat", "p0")),
                ("terminal-code", lambda row: row.__setitem__("terminal_code", "bad")),
                ("terminal-classification", lambda row: row.__setitem__("terminal_classification", "halted")),
                ("terminal-outcome", lambda row: row.__setitem__("terminal_outcome", "draw")),
                ("terminal-reward", lambda row: row.__setitem__("terminal_reward", [0, 0])),
                ("winner", lambda row: row.__setitem__("winner", "p1")),
                ("result", lambda row: row.__setitem__("candidate_result", "loss")),
                ("points", lambda row: row.__setitem__("candidate_half_points", 0)),
                ("decision-count", lambda row: row.__setitem__("decision_count", 0)),
                ("decision-routing", lambda row: row.__setitem__("candidate_decisions", 0)),
            )
            for name, mutate in game_mutations:
                target = clone(f"game-{name}")
                manifest = read_json_file(target / "run.json")
                games = _jsonl(target / "games.jsonl")
                mutate(games[0])
                _write_jsonl(target / "games.jsonl", games)
                _refresh_file_metadata(target, manifest, "games.jsonl")
                (target / "run.json").write_bytes(canonical_json_bytes(manifest))
                with self.subTest(game=name), self.assertRaises(ValueError):
                    validate_sampled_evaluation(target)

            pair_mutations = (
                ("schema", lambda row: row.__setitem__("schema", "bad")),
                ("pair", lambda row: row.__setitem__("pair_index", 1)),
                ("env", lambda row: row.__setitem__("env_seed", row["env_seed"] + 1)),
                ("p0-episode", lambda row: row.__setitem__("candidate_as_p0_episode_id", 2)),
                ("p1-episode", lambda row: row.__setitem__("candidate_as_p1_episode_id", 3)),
                ("p0-points", lambda row: row.__setitem__("candidate_as_p0_half_points", 0)),
                ("p1-points", lambda row: row.__setitem__("candidate_as_p1_half_points", 2)),
                ("total", lambda row: row.__setitem__("total_half_points", 1)),
            )
            for name, mutate in pair_mutations:
                target = clone(f"pair-{name}")
                manifest = read_json_file(target / "run.json")
                pairs = _jsonl(target / "pairs.jsonl")
                mutate(pairs[0])
                _write_jsonl(target / "pairs.jsonl", pairs)
                _refresh_file_metadata(target, manifest, "pairs.jsonl")
                (target / "run.json").write_bytes(canonical_json_bytes(manifest))
                with self.subTest(pair=name), self.assertRaises(ValueError):
                    validate_sampled_evaluation(target)

            noncanonical = clone("noncanonical")
            manifest = read_json_file(noncanonical / "run.json")
            (noncanonical / "run.json").write_text(json.dumps(manifest, indent=2), encoding="ascii")
            with self.assertRaises(ValueError):
                validate_sampled_evaluation(noncanonical)

    def test_same_head_forged_nonidentical_aa_legs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            pristine = root / "pristine"
            _evaluate_fixture(launcher, store, head, pristine, pairs=1)

            target = root / "forged-counts"
            shutil.copytree(pristine, target)
            manifest = read_json_file(target / "run.json")
            games = _jsonl(target / "games.jsonl")
            games[1]["candidate_decisions"] = 1
            games[1]["baseline_decisions"] = 0
            _write_jsonl(target / "games.jsonl", games)
            _refresh_file_metadata(target, manifest, "games.jsonl")
            (target / "run.json").write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "physical decision counts"):
                validate_sampled_evaluation(target)

            target = root / "forged-terminal"
            shutil.copytree(pristine, target)
            manifest = read_json_file(target / "run.json")
            games = _jsonl(target / "games.jsonl")
            pairs = _jsonl(target / "pairs.jsonl")
            games[1].update(
                {
                    "terminal_outcome": "p1_win",
                    "winner": "p1",
                    "terminal_reward": [-1, 1],
                    "candidate_result": "win",
                    "candidate_half_points": 2,
                }
            )
            points = PairedGamePoints(2, 2)
            pairs[0]["candidate_as_p1_half_points"] = 2
            pairs[0]["total_half_points"] = 4
            score = score_pair_half_points(
                [points.total_half_points],
                derive_evaluation_bootstrap_seed(manifest["configuration"]["base_seed"]),
                manifest["configuration"]["bootstrap_replicates"],
            )
            manifest["statistics"] = statistics_payload(score, summarize_paired_game_points([points]))
            _write_jsonl(target / "games.jsonl", games)
            _write_jsonl(target / "pairs.jsonl", pairs)
            _refresh_file_metadata(target, manifest, "games.jsonl")
            _refresh_file_metadata(target, manifest, "pairs.jsonl")
            (target / "run.json").write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "physical terminal"):
                validate_sampled_evaluation(target)


class SampledRealEnvironmentTest(unittest.TestCase):
    def test_real_environment_aa_then_ab_smoke(self) -> None:
        env_value = os.environ.get("MTG_KERNEL_RL_ENV_BIN")
        if not env_value:
            self.skipTest("MTG_KERNEL_RL_ENV_BIN not set")
        env_bin = Path(env_value)
        self.assertTrue(env_bin.is_file())
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
            update_zero = TrainingStore(store).validate_latest().head.head
            before_aa = _tree_bytes(store)
            aa = _evaluate_fixture(
                env_bin,
                store,
                update_zero,
                root / "aa",
                pairs=1,
                max_decisions=5_000,
                timeout_ms=60_000,
            )
            self.assertEqual((aa.total_half_points, aa.estimate), (2, 0.5))
            validate_sampled_evaluation(root / "aa")
            self.assertEqual(_tree_bytes(store), before_aa)

            train(env_bin=env_bin, out_dir=store, resume=store / "latest.json", until_update=1)
            update_one = TrainingStore(store).validate_latest().head.head
            self.assertNotEqual(update_one, update_zero)
            before_ab = _tree_bytes(store)
            ab = _evaluate_fixture(
                env_bin,
                store,
                update_one,
                root / "ab",
                pairs=1,
                max_decisions=5_000,
                timeout_ms=60_000,
            )
            self.assertEqual(ab.candidate_head, update_one)
            self.assertEqual(ab.baseline_head, update_zero)
            validate_sampled_evaluation(root / "ab")
            self.assertEqual(_tree_bytes(store), before_ab)


if __name__ == "__main__":
    unittest.main()
