from __future__ import annotations

import filecmp
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from mtg_kernel_rl.action_sampling import fixed_categorical_sampler_contract
from mtg_kernel_rl.client import Decision
from mtg_kernel_rl.rollout import (
    RUNNER_ACTION_SELECTION_CONTRACT,
    RUNNER_ARTIFACT_SCHEMA_VERSION,
    RUNNER_SEED_DERIVATION_CONTRACT,
    V1_RUNNER_ARTIFACT_SCHEMA_VERSION,
    run_episodes,
    select_action,
)
from mtg_kernel_rl.sampled_evaluation_store import ACTION_SELECTION_CONTRACT as EVALUATOR_ACTION_SELECTION_CONTRACT
from mtg_kernel_rl.training_store import TRAINER_ACTION_SELECTION_CONTRACT

from fixtures import DECK_HASHES, DECK_IDS, PROVENANCE, actor_observation, fake_launcher, legal_actions


class _FixedModelPolicy:
    def logits_value(self, _encoded):  # type: ignore[no-untyped-def]
        return torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)


class RolloutTest(unittest.TestCase):
    def test_deck_identity_failures_publish_no_runner_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            for scenario in ("deck_id_drift", "deck_hash_shape", "deck_hash_drift"):
                with self.subTest(scenario=scenario):
                    out = tmp / scenario
                    with self.assertRaises(Exception):
                        run_episodes(
                            env_bin=fake_launcher(tmp, scenario),
                            out_dir=out,
                            episodes=1,
                            base_seed=71_501,
                            max_decisions=8,
                            p0="uniform",
                            p1="uniform",
                        )
                    self.assertFalse(out.exists())

    def test_sampled_selector_has_fixed_goldens_and_preserves_global_rng(self) -> None:
        cases = (
            (0, 0, "p0"),
            (0, 0, "p1"),
            (1, 0, "p0"),
            (0, 1, "p0"),
        )
        random.seed(123_456)
        torch.manual_seed(234_567)
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        selected: list[int] = []
        for episode, step, actor in cases:
            decision = Decision(
                episode,
                step,
                actor,
                actor_observation(actor, step),
                legal_actions(actor),
                dict(PROVENANCE),
                DECK_IDS,
                DECK_HASHES,
            )
            selected.append(
                select_action(
                    decision,
                    policy="sampled",
                    base_seed=71_501,
                    episode=episode,
                    model_policy=_FixedModelPolicy(),  # type: ignore[arg-type]
                )
            )
        self.assertEqual(selected, [2, 2, 0, 2])
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_three_lanes_share_equal_nonaliased_categorical_contracts(self) -> None:
        cores = (
            EVALUATOR_ACTION_SELECTION_CONTRACT["categorical_sampler"],
            TRAINER_ACTION_SELECTION_CONTRACT["categorical_sampler"],
            RUNNER_ACTION_SELECTION_CONTRACT["sampled"]["categorical_sampler"],
        )
        expected = fixed_categorical_sampler_contract()
        for core in cores:
            self.assertEqual(core, expected)
        self.assertEqual(len({id(core) for core in cores}), 3)
        self.assertEqual(len({id(core["decimal_softmax"]) for core in cores}), 3)
        self.assertEqual(len({id(core["probability_mass"]) for core in cores}), 3)

        fresh = fixed_categorical_sampler_contract()
        fresh["decimal_softmax"]["exp_precision_digits"] = 1
        fresh["probability_mass"]["total"] = "2**1"
        for core in cores:
            self.assertEqual(core, expected)

    def test_fake_rollout_artifacts_are_byte_deterministic_and_terminal_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "valid")
            out_a = tmp / "a"
            out_b = tmp / "b"
            run_episodes(env_bin=launcher, out_dir=out_a, episodes=2, base_seed=71501, max_decisions=8, p0="uniform", p1="uniform")
            run_episodes(env_bin=launcher, out_dir=out_b, episodes=2, base_seed=71501, max_decisions=8, p0="uniform", p1="uniform")
            self.assertTrue(filecmp.cmp(out_a / "run.json", out_b / "run.json", shallow=False))
            self.assertTrue(filecmp.cmp(out_a / "episodes.jsonl", out_b / "episodes.jsonl", shallow=False))
            run_text = (out_a / "run.json").read_text(encoding="utf-8")
            episodes_text = (out_a / "episodes.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(str(out_a), run_text)
            self.assertNotIn("observation", run_text)
            self.assertNotIn("legal_actions", run_text)
            self.assertIn('"halted":0', run_text)
            self.assertIn('"truncated":0', run_text)
            self.assertIn('"terminal_outcome"', episodes_text)
            manifest_a = json.loads(run_text)
            episode_rows = [json.loads(line) for line in episodes_text.splitlines()]
            self.assertEqual(tuple(manifest_a["environment"]["deck_ids"]), DECK_IDS)
            self.assertEqual(tuple(manifest_a["environment"]["deck_hashes"]), DECK_HASHES)
            self.assertTrue(all(tuple(row["deck_ids"]) == DECK_IDS for row in episode_rows))
            self.assertTrue(all(tuple(row["deck_hashes"]) == DECK_HASHES for row in episode_rows))
            manifest = run_episodes(
                env_bin=launcher,
                out_dir=tmp / "mixed",
                episodes=2,
                base_seed=71_502,
                max_decisions=8,
                p0="uniform",
                p1="greedy",
            )
            self.assertEqual(
                manifest["action_selection"],
                {
                    "p0": RUNNER_ACTION_SELECTION_CONTRACT["uniform"],
                    "p1": RUNNER_ACTION_SELECTION_CONTRACT["greedy"],
                },
            )

    def test_sampled_rollout_v3_contract_is_repeatable_and_legacy_versions_remain_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "valid")
            out_a = tmp / "sampled-a"
            out_b = tmp / "sampled-b"
            manifest_a = run_episodes(
                env_bin=launcher,
                out_dir=out_a,
                episodes=4,
                base_seed=71_501,
                max_decisions=8,
                p0="sampled",
                p1="sampled",
            )
            manifest_b = run_episodes(
                env_bin=launcher,
                out_dir=out_b,
                episodes=4,
                base_seed=71_501,
                max_decisions=8,
                p0="sampled",
                p1="sampled",
            )
            self.assertEqual(manifest_a, manifest_b)
            self.assertTrue(filecmp.cmp(out_a / "run.json", out_b / "run.json", shallow=False))
            self.assertTrue(filecmp.cmp(out_a / "episodes.jsonl", out_b / "episodes.jsonl", shallow=False))
            self.assertEqual(manifest_a["artifact_schema_version"], RUNNER_ARTIFACT_SCHEMA_VERSION)
            self.assertEqual(RUNNER_ARTIFACT_SCHEMA_VERSION, 3)
            self.assertEqual(
                manifest_a["action_selection"],
                {
                    "p0": RUNNER_ACTION_SELECTION_CONTRACT["sampled"],
                    "p1": RUNNER_ACTION_SELECTION_CONTRACT["sampled"],
                },
            )
            self.assertEqual(manifest_a["seed_derivation"], RUNNER_SEED_DERIVATION_CONTRACT)
            self.assertEqual(V1_RUNNER_ARTIFACT_SCHEMA_VERSION, 1)
            self.assertNotEqual(manifest_a["artifact_schema_version"], V1_RUNNER_ARTIFACT_SCHEMA_VERSION)
            self.assertNotEqual(manifest_a["artifact_schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
