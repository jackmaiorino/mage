from __future__ import annotations

import random
import unittest

import torch

from mtg_kernel_rl.determinism import (
    EVALUATOR_SEED_DERIVATION_VERSION,
    TRAINER_SEED_DERIVATION_VERSION,
    EvaluatorSeedDerivation,
    TrainerSeedDerivation,
    configure_torch_determinism,
    derive_evaluation_bootstrap_seed,
    derive_evaluation_env_seed,
    derive_model_init_seed,
    derive_train_env_seed,
    derive_train_learner_action_seed,
    derive_train_opponent_action_seed,
)
from mtg_kernel_rl.features import encode_decision
from mtg_kernel_rl.model import INITIALIZER_RUNNER_FIXED_V1, INITIALIZER_TRAINER_SEEDED_V1, KernelPolicyValueNet

from fixtures import legal_actions, observation


class TrainerDeterminismTest(unittest.TestCase):
    def test_sha256_evaluator_seed_known_vectors_and_frozen_contract(self) -> None:
        self.assertEqual(EVALUATOR_SEED_DERIVATION_VERSION, "kernel-python-rl-evaluator-sha256-v1")
        self.assertEqual(derive_evaluation_bootstrap_seed(71501), 0x4389_D43A_6139_A202)
        self.assertEqual(derive_evaluation_env_seed(71501, 0), 0x37E6_5017_E207_5B88)
        self.assertEqual(derive_evaluation_env_seed(71501, 1), 0x5792_DDBA_A6C5_5F43)
        self.assertEqual(derive_evaluation_env_seed(71501, 32), 0x5139_F8A6_C14F_1136)
        self.assertEqual(
            EvaluatorSeedDerivation().namespaces,
            (
                "evaluation-bootstrap/base_seed",
                "evaluation-env/base_seed/pair_index",
            ),
        )
        self.assertEqual(TRAINER_SEED_DERIVATION_VERSION, "kernel-python-rl-trainer-sha256-v1")
        self.assertEqual(
            TrainerSeedDerivation().namespaces,
            (
                "model-init/base_seed",
                "train-env/base_seed/pair_index",
                "train-learner-action/base_seed/episode_index/learner_decision_index",
                "train-opponent-action/base_seed/episode_index/opponent_decision_index",
            ),
        )
        self.assertNotEqual(derive_evaluation_env_seed(71501, 0), derive_train_env_seed(71501, 0))
        for bad in (True, -1, 2**63):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                derive_evaluation_bootstrap_seed(bad)  # type: ignore[arg-type]
            with self.subTest(pair_index=bad), self.assertRaises((TypeError, ValueError)):
                derive_evaluation_env_seed(71501, bad)  # type: ignore[arg-type]

    def test_sha256_trainer_seed_known_vectors_and_separation(self) -> None:
        self.assertEqual(derive_model_init_seed(71501), 9076772781811365075)
        self.assertEqual(derive_train_env_seed(71501, 0), 7253935443031715823)
        self.assertEqual(derive_train_env_seed(71501, 1), 7044699237811831443)
        self.assertEqual(derive_train_learner_action_seed(71501, 2, 3), 7877844131612960500)
        self.assertEqual(derive_train_opponent_action_seed(71501, 2, 3), 2429204417625999091)
        self.assertEqual(derive_train_env_seed(71501, 0), derive_train_env_seed(71501, 0))
        self.assertNotEqual(derive_train_env_seed(71501, 0), derive_train_env_seed(71501, 1))
        self.assertNotEqual(derive_train_env_seed(71501, 3), derive_train_learner_action_seed(71501, 3, 0))
        self.assertNotEqual(derive_train_env_seed(1, 256), derive_train_env_seed(256, 1))
        for bad in (True, -1, 2**63):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                derive_model_init_seed(bad)  # type: ignore[arg-type]

    def test_paired_env_seeds_and_actor_local_action_counters(self) -> None:
        seeds = [derive_train_env_seed(71501, episode // 2) for episode in range(6)]
        self.assertEqual(seeds[0], seeds[1])
        self.assertEqual(seeds[2], seeds[3])
        self.assertEqual(seeds[4], seeds[5])
        self.assertNotEqual(seeds[1], seeds[2])
        self.assertEqual(
            derive_train_learner_action_seed(71501, 4, 0),
            derive_train_learner_action_seed(71501, 4, 0),
        )
        self.assertNotEqual(
            derive_train_learner_action_seed(71501, 4, 0),
            derive_train_learner_action_seed(71501, 4, 1),
        )

    def test_seeded_model_init_repeats_differs_and_ignores_global_rng(self) -> None:
        configure_torch_determinism()
        encoded = encode_decision(observation(), legal_actions())
        config = KernelPolicyValueNet.from_encoded(encoded).config
        seed = derive_model_init_seed(71501)
        torch.manual_seed(999)
        random.seed(999)
        model_a = KernelPolicyValueNet(config, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=seed)
        torch.rand(17)
        random.random()
        model_b = KernelPolicyValueNet(config, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=seed)
        model_c = KernelPolicyValueNet(config, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=seed + 1)
        for key in model_a.state_dict():
            self.assertTrue(torch.equal(model_a.state_dict()[key], model_b.state_dict()[key]))
        self.assertTrue(any(not torch.equal(model_a.state_dict()[key], model_c.state_dict()[key]) for key in model_a.state_dict()))
        fixed_a = KernelPolicyValueNet.from_encoded(encoded)
        fixed_b = KernelPolicyValueNet(config, initializer=INITIALIZER_RUNNER_FIXED_V1)
        for key in fixed_a.state_dict():
            self.assertTrue(torch.equal(fixed_a.state_dict()[key], fixed_b.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
