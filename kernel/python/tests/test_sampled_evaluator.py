from __future__ import annotations

import contextlib
import decimal
import importlib
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

import mtg_kernel_rl.action_sampling as action_sampling_mod
import mtg_kernel_rl.cli as cli_mod
import mtg_kernel_rl.sampled_evaluator as sampled_mod
import mtg_kernel_rl.sampled_evaluation_store as sampled_store_mod
from mtg_kernel_rl.artifact_io import canonical_json_bytes, read_json_file, sha256_bytes
from mtg_kernel_rl.artifacts import set_fault_injector
from mtg_kernel_rl.client import Decision, Terminal
from mtg_kernel_rl.determinism import derive_evaluation_action_seed, derive_evaluation_bootstrap_seed
from mtg_kernel_rl.evaluation_stats import PairedGamePoints, score_pair_half_points, summarize_paired_game_points
from mtg_kernel_rl.evaluation_store import statistics_payload, validate_evaluation
from mtg_kernel_rl.evaluator import EvaluationResult, evaluate
from mtg_kernel_rl.output_lock import OutputLock
from mtg_kernel_rl.path_safety import OUTPUT_LOCK_FILE_NAME
from mtg_kernel_rl.sampled_evaluation_store import (
    ACTION_SEED_DERIVATION_CONTRACT,
    ACTION_SELECTION_CONTRACT,
    ALGORITHM_CONTRACT,
    GAME_SCHEMA,
    PAIR_SCHEMA,
    RUN_SCHEMA,
    SEAT_SCHEDULE_CONTRACT,
    V2_ACTION_SELECTION_CONTRACT,
    V2_ALGORITHM_CONTRACT,
    V2_GAME_SCHEMA,
    V2_PAIR_SCHEMA,
    V2_RUN_SCHEMA,
    V3_ACTION_SELECTION_CONTRACT,
    V3_ALGORITHM_CONTRACT,
    V3_GAME_SCHEMA,
    V3_PAIR_SCHEMA,
    V3_RUN_SCHEMA,
    validate_sampled_evaluation,
)
from mtg_kernel_rl.sampled_evaluator import _run_sampled_game, _select_sampled_action, evaluate_sampled
from mtg_kernel_rl.trainer import train
from mtg_kernel_rl.training_store import TrainingStore

from fixtures import DECK_HASHES, DECK_IDS, PROVENANCE, actor_observation, fake_launcher, legal_actions


V3_ACTION_SELECTION_GOLDEN = {
    "categorical_sampler": {
        "action_rng": "one splitmix64-v1 uint64 output per decision, initialized directly from the action seed",
        "algorithm": "inverse CDF over Hamilton-apportioned 2**64-unit mass in legal-action order",
        "decimal_softmax": {
            "context": {
                "capitals": 1,
                "clamp": 0,
                "emax": 999_999,
                "emin": -999_999,
                "flags_initially_set": [],
                "traps": ["InvalidOperation", "DivisionByZero", "Overflow"],
            },
            "delta_precision_digits": 256,
            "exp_cutoff": "strictly below -128 receives zero mass",
            "exp_precision_digits": 80,
            "input": "exact IEEE-754 binary32 logits converted to Decimal",
            "rounding": "ROUND_HALF_EVEN",
        },
        "probability_mass": {
            "apportionment": (
                "floor exact normalized Decimal-exp shares, then residual units by descending exact remainder "
                "and ascending legal-action index"
            ),
            "total": "2**64",
        },
        "sampler_version": "decimal-softmax-hamilton-splitmix64-v1",
    },
    "inference": "torch.inference_mode",
    "mode": "sampled_softmax",
    "replacement": False,
    "temperature_hex": "0x1.0000000000000p+0",
}


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


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {name: (root / name).read_bytes() for name in ("games.jsonl", "pairs.jsonl", "run.json")}


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


_HARD_EXIT_SAMPLED_EVALUATOR = r"""
import os
import sys
from pathlib import Path

from mtg_kernel_rl.artifacts import set_fault_injector
from mtg_kernel_rl.sampled_evaluator import evaluate_sampled

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
evaluate_sampled(
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


class SampledSelectorTest(unittest.TestCase):
    def _decision(self) -> Decision:
        return Decision(
            0,
            0,
            "p0",
            actor_observation("p0"),
            legal_actions("p0"),
            dict(PROVENANCE),
            DECK_IDS,
            DECK_HASHES,
        )

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
        self.assertEqual(selected, [1, 2, 1, 1, 2, 2])
        self.assertEqual(_select_sampled_action(FixedModel(), decision, seeds[0]), selected[0])
        self.assertGreater(len(set(selected)), 1)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_selector_has_frozen_softmax_rng_and_boundary_vectors(self) -> None:
        scale = 1 << 64
        logits = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        expected_weights = (
            1_660_770_942_083_389_871,
            4_514_443_473_098_088_136,
            12_271_529_658_528_073_609,
        )
        self.assertEqual(action_sampling_mod.fixed_softmax_mass(logits), expected_weights)
        self.assertEqual(
            action_sampling_mod.fixed_softmax_mass(torch.tensor([100.0, 101.0, 102.0], dtype=torch.float32)),
            action_sampling_mod.fixed_softmax_mass(logits),
        )
        self.assertEqual(
            action_sampling_mod.fixed_softmax_mass(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)),
            (6_148_914_691_236_517_206, 6_148_914_691_236_517_205, 6_148_914_691_236_517_205),
        )
        self.assertEqual(
            action_sampling_mod.fixed_softmax_mass(torch.tensor([0.0, -128.0, -129.0], dtype=torch.float32)),
            (scale, 0, 0),
        )
        float32_max = torch.finfo(torch.float32).max
        self.assertEqual(
            action_sampling_mod.fixed_softmax_mass(torch.tensor([float32_max, -float32_max], dtype=torch.float32)),
            (scale, 0),
        )
        self.assertEqual(
            [action_sampling_mod.splitmix64_u64(seed) for seed in (0, 1, (1 << 63) - 1)],
            [0xE220_A839_7B1D_CDAF, 0x910A_2DEC_8902_5CC1, 0x2A67_D755_2E03_9EA7],
        )
        boundary_weights = (2, 3, scale - 5)
        self.assertEqual(
            [
                action_sampling_mod.select_categorical_u64(boundary_weights, draw)
                for draw in (0, 1, 2, 4, 5, scale - 1)
            ],
            [0, 0, 1, 1, 2, 2],
        )
        with mock.patch.object(torch, "softmax", side_effect=AssertionError("Torch softmax must not be used")), mock.patch.object(
            torch,
            "multinomial",
            side_effect=AssertionError("Torch multinomial must not be used"),
        ):
            self.assertEqual(action_sampling_mod.sample_fixed_categorical(logits, 0), 2)

        original_context = decimal.getcontext().copy()
        try:
            decimal.getcontext().prec = 3
            decimal.getcontext().rounding = decimal.ROUND_DOWN
            self.assertEqual(action_sampling_mod.fixed_softmax_mass(logits), expected_weights)
            self.assertEqual((decimal.getcontext().prec, decimal.getcontext().rounding), (3, decimal.ROUND_DOWN))
        finally:
            decimal.setcontext(original_context)

    def test_selector_context_is_frozen_before_module_import(self) -> None:
        script = r"""
import decimal
decimal.DefaultContext.prec = 3
decimal.DefaultContext.rounding = decimal.ROUND_DOWN
decimal.DefaultContext.Emin = -9
decimal.DefaultContext.Emax = 9
decimal.DefaultContext.capitals = 0
decimal.DefaultContext.clamp = 1
for signal in decimal.DefaultContext.traps:
    decimal.DefaultContext.traps[signal] = True

import torch
from mtg_kernel_rl.action_sampling import fixed_softmax_mass

print(",".join(str(value) for value in fixed_softmax_mass(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32))))
"""
        env = os.environ.copy()
        python_root = str(Path(action_sampling_mod.__file__).resolve().parents[1])
        env["PYTHONPATH"] = python_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "1660770942083389871,4514443473098088136,12271529658528073609",
        )

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
        with self.assertRaisesRegex(ValueError, r"sum to 2\*\*64"):
            action_sampling_mod.select_categorical_u64((1, 2, 3), 0)
        with mock.patch.object(sampled_mod, "sample_fixed_categorical", return_value=3):
            with self.assertRaisesRegex(ValueError, "out-of-range legal action"):
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
                    DECK_IDS,
                    DECK_HASHES,
                )

            def reset(
                self,
                *,
                episode_id: int,
                env_seed: int,
                max_decisions: int,
                deck_ids: tuple[str, str],
            ):  # type: ignore[no-untyped-def]
                if deck_ids != DECK_IDS:
                    raise AssertionError("sampled evaluator changed physical deck order")
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
                    DECK_IDS,
                    DECK_HASHES,
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
                deck_ids=DECK_IDS,
                deck_hashes=DECK_HASHES,
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
                deck_ids=DECK_IDS,
                deck_hashes=DECK_HASHES,
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


class SampledV3FreezeTest(unittest.TestCase):
    def test_v3_action_contract_is_its_own_golden_under_future_live_sampler_change(self) -> None:
        self.assertEqual(V3_ACTION_SELECTION_CONTRACT, V3_ACTION_SELECTION_GOLDEN)
        self.assertEqual(ACTION_SELECTION_CONTRACT, V3_ACTION_SELECTION_GOLDEN)
        future_sampler = action_sampling_mod.fixed_categorical_sampler_contract()
        future_sampler["sampler_version"] = "simulated-future-sampler-v99"
        self.assertNotEqual(future_sampler, V3_ACTION_SELECTION_GOLDEN["categorical_sampler"])

        try:
            with mock.patch.object(
                action_sampling_mod,
                "fixed_categorical_sampler_contract",
                return_value=future_sampler,
            ):
                reloaded = importlib.reload(sampled_store_mod)
                self.assertEqual(reloaded.V3_ACTION_SELECTION_CONTRACT, V3_ACTION_SELECTION_GOLDEN)
                self.assertEqual(reloaded.ACTION_SELECTION_CONTRACT["categorical_sampler"], future_sampler)
                self.assertNotEqual(
                    reloaded.V3_ACTION_SELECTION_CONTRACT,
                    reloaded.ACTION_SELECTION_CONTRACT,
                )
        finally:
            importlib.reload(sampled_store_mod)


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
            self.assertEqual(V3_ALGORITHM_CONTRACT, ALGORITHM_CONTRACT)
            self.assertEqual(V3_ACTION_SELECTION_CONTRACT, V3_ACTION_SELECTION_GOLDEN)
            self.assertEqual(manifest["action_seed_derivation"], ACTION_SEED_DERIVATION_CONTRACT)
            self.assertEqual(manifest["action_selection"], ACTION_SELECTION_CONTRACT)
            self.assertEqual(manifest["seat_schedule"], SEAT_SCHEDULE_CONTRACT)
            games = _jsonl(root / "sampled-a" / "games.jsonl")
            pairs = _jsonl(root / "sampled-a" / "pairs.jsonl")
            self.assertTrue(all(len(row) == 19 and row["schema"] == GAME_SCHEMA for row in games))
            self.assertTrue(all(len(row) == 10 and row["schema"] == PAIR_SCHEMA for row in pairs))
            self.assertTrue(all(tuple(row["deck_ids"]) == DECK_IDS for row in games + pairs))
            self.assertTrue(all(tuple(row["deck_hashes"]) == DECK_HASHES for row in games + pairs))
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
            "--deck-ids",
            "Burn",
            "Rally",
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
        self.assertEqual(called.call_args.kwargs["deck_ids"], ("Burn", "Rally"))
        line = output.getvalue()
        self.assertEqual(line, json.dumps(json.loads(line), sort_keys=True, separators=(",", ":")) + "\n")
        self.assertEqual(
            set(json.loads(line)),
            {"baseline_head", "candidate_head", "estimate_hex", "game_count", "pair_count", "run_sha256", "total_half_points"},
        )


class SampledPublicationProofTest(unittest.TestCase):
    def test_data_file_faults_are_uncommitted_and_run_fault_is_postcommit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            source_before = _tree_bytes(store)
            cases = (
                ("games.jsonl", {OUTPUT_LOCK_FILE_NAME, "games.jsonl"}, False),
                ("pairs.jsonl", {OUTPUT_LOCK_FILE_NAME, "games.jsonl", "pairs.jsonl"}, False),
                ("run.json", {OUTPUT_LOCK_FILE_NAME, "games.jsonl", "pairs.jsonl", "run.json"}, True),
            )
            for target_name, expected_entries, committed in cases:
                with self.subTest(target=target_name):
                    out = root / f"sampled-fault-{target_name.replace('.', '-')}"
                    fired = {"value": False}

                    def injector(boundary: str, path: Path | None) -> None:
                        if (
                            not fired["value"]
                            and boundary == "json_replace_after"
                            and path is not None
                            and path.name == target_name
                        ):
                            fired["value"] = True
                            raise RuntimeError(f"sampled fault after {target_name}")

                    previous = set_fault_injector(injector)
                    try:
                        with self.assertRaisesRegex(RuntimeError, target_name.replace(".", r"\.")):
                            _evaluate_fixture(launcher, store, head, out, pairs=1)
                    finally:
                        set_fault_injector(previous)
                    self.assertTrue(fired["value"])
                    self.assertEqual({path.name for path in out.iterdir()}, expected_entries)
                    if committed:
                        validated = validate_sampled_evaluation(out)
                        self.assertEqual((validated.total_half_points, validated.estimate), (2, 0.5))
                    else:
                        self.assertFalse((out / "run.json").exists())
                        with self.assertRaises(ValueError):
                            validate_sampled_evaluation(out)
                    self.assertEqual(_tree_bytes(store), source_before)
                    output_before_retry = _tree_bytes(out)
                    with self.assertRaises(FileExistsError):
                        _evaluate_fixture(launcher, store, head, out, pairs=1)
                    self.assertEqual(_tree_bytes(out), output_before_retry)
                    self.assertEqual(_tree_bytes(store), source_before)

    def test_hard_exit_at_every_atomic_replace_boundary_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            source_before = _tree_bytes(store)
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
                        slug = f"sampled-{boundary}-{target_name.replace('.', '-')}"
                        out = root / slug
                        marker = root / f"{slug}.marker"
                        completed = subprocess.run(
                            [
                                sys.executable,
                                "-c",
                                _HARD_EXIT_SAMPLED_EVALUATOR,
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

                        with OutputLock(out):
                            pass
                        if boundary == "json_replace_after" and target_name == "run.json":
                            validated = validate_sampled_evaluation(out)
                            self.assertEqual((validated.total_half_points, validated.estimate), (2, 0.5))
                        else:
                            with self.assertRaises(ValueError):
                                validate_sampled_evaluation(out)
                        self.assertEqual(_tree_bytes(store), source_before)
                        output_before_retry = _tree_bytes(out)
                        with self.assertRaises(FileExistsError):
                            _evaluate_fixture(launcher, store, head, out, pairs=1)
                        self.assertEqual(_tree_bytes(out), output_before_retry)
                        self.assertEqual(_tree_bytes(store), source_before)

    def test_runtime_failure_leaves_lock_only_and_same_root_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            scenario_file = root / "scenario.txt"
            scenario_file.write_text("valid", encoding="utf-8")
            launcher, store, head = _train_fixture(
                root,
                scenario="switchable",
                extra_env={"FAKE_SCENARIO_FILE": str(scenario_file)},
            )
            out = root / "sampled-retry"
            scenario_file.write_text("train_late_fault", encoding="utf-8")
            with self.assertRaises(Exception):
                _evaluate_fixture(launcher, store, head, out, pairs=1)
            self.assertEqual({path.name for path in out.iterdir()}, {OUTPUT_LOCK_FILE_NAME})

            scenario_file.write_text("valid", encoding="utf-8")
            result = _evaluate_fixture(launcher, store, head, out, pairs=1)
            self.assertEqual((result.total_half_points, result.estimate), (2, 0.5))
            validate_sampled_evaluation(out)

    def test_fresh_output_root_accepts_new_or_empty_and_rejects_dirty_or_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)

            new_root = root / "new-root"
            _evaluate_fixture(launcher, store, head, new_root, pairs=1)
            validate_sampled_evaluation(new_root)

            empty_root = root / "empty-root"
            empty_root.mkdir()
            _evaluate_fixture(launcher, store, head, empty_root, pairs=1)
            validate_sampled_evaluation(empty_root)

            dirty_root = root / "dirty-root"
            dirty_root.mkdir()
            sentinel = dirty_root / "sentinel.txt"
            sentinel.write_bytes(b"preserve-me")
            with self.assertRaises(FileExistsError):
                _evaluate_fixture(launcher, store, head, dirty_root, pairs=1)
            self.assertEqual(sentinel.read_bytes(), b"preserve-me")
            self.assertEqual({path.name for path in dirty_root.iterdir()}, {OUTPUT_LOCK_FILE_NAME, "sentinel.txt"})

            before = _tree_bytes(new_root)
            with self.assertRaises(FileExistsError):
                _evaluate_fixture(launcher, store, head, new_root, pairs=1)
            self.assertEqual(_tree_bytes(new_root), before)

            file_root = root / "file-root"
            file_root.write_bytes(b"not-a-directory")
            with self.assertRaises((FileExistsError, ValueError)):
                _evaluate_fixture(launcher, store, head, file_root, pairs=1)
            self.assertEqual(file_root.read_bytes(), b"not-a-directory")


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
                ("legacy-v1-run-schema", lambda value: value.__setitem__("schema", "kernel_rl_paired_evaluation/v1")),
                ("legacy-v2-run-schema", lambda value: value.__setitem__("schema", V2_RUN_SCHEMA)),
                ("legacy-v3-run-schema", lambda value: value.__setitem__("schema", V3_RUN_SCHEMA)),
                (
                    "legacy-v2-artifact-schemas",
                    lambda value: value.__setitem__(
                        "artifact_schemas",
                        {"game": V2_GAME_SCHEMA, "pair": V2_PAIR_SCHEMA, "run": V2_RUN_SCHEMA},
                    ),
                ),
                (
                    "legacy-v3-artifact-schemas",
                    lambda value: value.__setitem__(
                        "artifact_schemas",
                        {"game": V3_GAME_SCHEMA, "pair": V3_PAIR_SCHEMA, "run": V3_RUN_SCHEMA},
                    ),
                ),
                ("legacy-v2-algorithm", lambda value: value.__setitem__("algorithm", V2_ALGORITHM_CONTRACT)),
                (
                    "legacy-v2-action-selection",
                    lambda value: value.__setitem__("action_selection", V2_ACTION_SELECTION_CONTRACT),
                ),
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
                ("selection-rng", lambda value: value["action_selection"]["categorical_sampler"].__setitem__("action_rng", "bad")),
                ("selection-algorithm", lambda value: value["action_selection"]["categorical_sampler"].__setitem__("algorithm", "bad")),
                ("selection-version", lambda value: value["action_selection"]["categorical_sampler"].__setitem__("sampler_version", "bad")),
                (
                    "selection-decimal",
                    lambda value: value["action_selection"]["categorical_sampler"]["decimal_softmax"].__setitem__("exp_precision_digits", 79),
                ),
                (
                    "selection-mass",
                    lambda value: value["action_selection"]["categorical_sampler"]["probability_mass"].__setitem__("total", "2**63"),
                ),
                ("seat-candidate-p0", lambda value: value["seat_schedule"].__setitem__("candidate_as_p0", "bad")),
                ("seat-candidate-p1", lambda value: value["seat_schedule"].__setitem__("candidate_as_p1", "bad")),
                ("seat-deck-order", lambda value: value["seat_schedule"].__setitem__("deck_order", "bad")),
                ("seat-env", lambda value: value["seat_schedule"].__setitem__("paired_environment_seed", "bad")),
                ("seat-action", lambda value: value["seat_schedule"].__setitem__("paired_physical_action_streams", "bad")),
                ("environment", lambda value: value["environment"].__setitem__("binary_sha256", "0" * 64)),
                ("environment-deck-hash", lambda value: value["environment"]["deck_hashes"].__setitem__(0, 1)),
                ("configuration-deck-id", lambda value: value["configuration"]["deck_ids"].__setitem__(1, "Rally")),
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
                ("deck-id", lambda row: row["deck_ids"].__setitem__(0, "Rally")),
                ("deck-hash", lambda row: row["deck_hashes"].__setitem__(0, 1)),
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
                ("deck-id", lambda row: row["deck_ids"].__setitem__(0, "Rally")),
                ("deck-hash", lambda row: row["deck_hashes"].__setitem__(0, 1)),
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


class SampledFilesystemBoundaryTest(unittest.TestCase):
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
            pristine = root / "pristine-symlink"
            _evaluate_fixture(launcher, store, head, pristine, pairs=1)
            for name in (OUTPUT_LOCK_FILE_NAME, "games.jsonl", "pairs.jsonl", "run.json"):
                with self.subTest(name=name):
                    target = root / f"sampled-symlink-{name.replace('.', '-')}"
                    shutil.copytree(pristine, target)
                    artifact = target / name
                    sentinel = outside / f"{name.replace('.', '-')}.sentinel"
                    sentinel.write_bytes(artifact.read_bytes())
                    expected = sentinel.read_bytes()
                    artifact.unlink()
                    os.symlink(sentinel, artifact)
                    with self.assertRaises(ValueError):
                        validate_sampled_evaluation(target)
                    self.assertEqual(sentinel.read_bytes(), expected)

    def test_committed_verifier_rejects_hardlinks_and_preserves_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            outside = root / "outside"
            outside.mkdir()
            launcher, store, head = _train_fixture(root)
            pristine = root / "pristine-hardlink"
            _evaluate_fixture(launcher, store, head, pristine, pairs=1)
            for name in (OUTPUT_LOCK_FILE_NAME, "games.jsonl", "pairs.jsonl", "run.json"):
                with self.subTest(name=name):
                    target = root / f"sampled-hardlink-{name.replace('.', '-')}"
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
                        validate_sampled_evaluation(target)
                    self.assertEqual(sentinel.read_bytes(), expected)

    def test_root_junction_or_symlink_is_rejected_for_verification_and_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            pristine = root / "pristine-root-alias"
            _evaluate_fixture(launcher, store, head, pristine, pairs=1)
            before = _artifact_bytes(pristine)
            verify_alias = root / "verify-root-alias"
            _create_directory_alias(self, verify_alias, pristine)
            try:
                with self.assertRaises(ValueError):
                    validate_sampled_evaluation(verify_alias)
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
                    _evaluate_fixture(launcher, store, head, publish_alias, pairs=1)
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
                result_root = alias_parent / "sampled-evaluation"
                _evaluate_fixture(launcher, store, head, result_root, pairs=1)
                validated = validate_sampled_evaluation(real_parent / "sampled-evaluation")
                self.assertEqual((validated.total_half_points, validated.estimate), (2, 0.5))
            finally:
                _remove_directory_alias(alias_parent)

    def test_exact_root_types_and_preexisting_aliases_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            pristine = root / "pristine-exact-root"
            _evaluate_fixture(launcher, store, head, pristine, pairs=1)

            wrong_type = root / "wrong-type"
            shutil.copytree(pristine, wrong_type)
            (wrong_type / "games.jsonl").unlink()
            (wrong_type / "games.jsonl").mkdir()
            with self.assertRaises(ValueError):
                validate_sampled_evaluation(wrong_type)

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
                _evaluate_fixture(launcher, store, head, hardlink_root, pairs=1)
            self.assertEqual(sentinel.read_bytes(), b"do-not-change")

            directory_root = root / "publish-directory"
            (directory_root / "run.json").mkdir(parents=True)
            nested_sentinel = directory_root / "run.json" / "sentinel.txt"
            nested_sentinel.write_bytes(b"nested-sentinel")
            with self.assertRaises(FileExistsError):
                _evaluate_fixture(launcher, store, head, directory_root, pairs=1)
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
                _evaluate_fixture(launcher, store, head, out, pairs=1)
            self.assertEqual(sentinel.read_bytes(), b"external-sentinel")

    def test_child_junction_or_directory_symlink_is_rejected_without_following(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            launcher, store, head = _train_fixture(root)
            pristine = root / "pristine-child-alias"
            _evaluate_fixture(launcher, store, head, pristine, pairs=1)
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
                    validate_sampled_evaluation(target)
                self.assertEqual(sentinel.read_bytes(), b"external-child-sentinel")
            finally:
                _remove_directory_alias(alias)


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
