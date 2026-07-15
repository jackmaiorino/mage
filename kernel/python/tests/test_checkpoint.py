from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from mtg_kernel_rl.checkpoint import (
    build_checkpoint_payload,
    create_adam,
    export_adam_state,
    load_adam_state,
    load_checkpoint_file,
    logical_state_hash,
    save_checkpoint_file,
    validate_checkpoint_payload,
    validate_model_state,
)
from mtg_kernel_rl.determinism import TrainerSeedDerivation, configure_torch_determinism, derive_model_init_seed
from mtg_kernel_rl.features import encode_decision
from mtg_kernel_rl.model import INITIALIZER_TRAINER_SEEDED_V1, KernelPolicyValueNet
from mtg_kernel_rl.trainer import _compatibility_tuple

from fixtures import legal_actions, observation


class CheckpointTest(unittest.TestCase):
    def make_payload(self) -> tuple[dict, KernelPolicyValueNet, torch.optim.Adam, dict]:
        configure_torch_determinism()
        encoded = encode_decision(observation(), legal_actions())
        config = KernelPolicyValueNet.from_encoded(encoded).config
        model = KernelPolicyValueNet(config, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=derive_model_init_seed(71501))
        optimizer = create_adam(model, 0.001)
        logits, value = model(encoded)
        loss = logits.sum() * 0.01 + value.square()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        compatibility = _compatibility_tuple()
        payload = build_checkpoint_payload(
            run_digest="r" * 64,
            completed_update=1,
            optimizer_step_count=1,
            next_episode=2,
            outcomes_by_learner_seat={"p0": {"win": 1, "loss": 0, "draw": 0}, "p1": {"win": 0, "loss": 1, "draw": 0}},
            learner_decisions_by_seat={"p0": 1, "p1": 1},
            model=model,
            optimizer=optimizer,
            learning_rate=0.001,
            base_seed=71501,
            seed_derivation={"schema": "test", "namespaces": list(TrainerSeedDerivation().namespaces)},
            provenance={"protocol": "kernel_rl_jsonl", "protocol_version": 2, "schema_version": 2, "kernel_version": "0.0.1-spike", "surface_version": 2, "card_db_hash": 1},
            compatibility=compatibility,
        )
        return payload, model, optimizer, compatibility

    def test_checkpoint_roundtrip_restores_model_optimizer_and_rng(self) -> None:
        payload, model, optimizer, compatibility = self.make_payload()
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "checkpoint.pt"
            save_checkpoint_file(path, payload)
            loaded = load_checkpoint_file(path)
        validate_checkpoint_payload(loaded, run_digest="r" * 64, compatibility=compatibility)
        restored = KernelPolicyValueNet(model.config, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=0)
        restored.load_state_dict(validate_model_state(restored, loaded["model_state"]), strict=True)
        restored_optimizer = create_adam(restored, 0.001)
        load_adam_state(restored_optimizer, restored, loaded["optimizer_state"], 0.001)
        for key in model.state_dict():
            self.assertTrue(torch.equal(model.state_dict()[key], restored.state_dict()[key]), key)
        self.assertEqual(export_adam_state(optimizer, model, 0.001)["config"], export_adam_state(restored_optimizer, restored, 0.001)["config"])
        for name, slot in export_adam_state(optimizer, model, 0.001)["state"].items():
            restored_slot = export_adam_state(restored_optimizer, restored, 0.001)["state"][name]
            self.assertEqual(set(slot), set(restored_slot))
            for key in slot:
                self.assertTrue(torch.equal(slot[key], restored_slot[key]), f"{name}.{key}")

    def test_logical_digest_changes_for_tensor_scalar_shape_dtype_and_rng_mutations(self) -> None:
        payload, _model, _optimizer, _compat = self.make_payload()
        base = logical_state_hash(payload)
        mutated_byte = copy.deepcopy(payload)
        first_key = next(iter(mutated_byte["model_state"]))
        mutated_byte["model_state"][first_key] = mutated_byte["model_state"][first_key].clone()
        mutated_byte["model_state"][first_key].view(-1)[0] += 1.0
        self.assertNotEqual(base, logical_state_hash(mutated_byte))
        mutated_shape = copy.deepcopy(payload)
        mutated_shape["model_state"][first_key] = mutated_shape["model_state"][first_key].reshape(-1)
        self.assertNotEqual(base, logical_state_hash(mutated_shape))
        mutated_dtype = copy.deepcopy(payload)
        mutated_dtype["model_state"][first_key] = mutated_dtype["model_state"][first_key].double()
        self.assertNotEqual(base, logical_state_hash(mutated_dtype))
        mutated_scalar = copy.deepcopy(payload)
        mutated_scalar["next_episode"] += 2
        self.assertNotEqual(base, logical_state_hash(mutated_scalar))
        mutated_rng = copy.deepcopy(payload)
        mutated_rng["torch_cpu_rng_state"] = mutated_rng["torch_cpu_rng_state"].clone()
        mutated_rng["torch_cpu_rng_state"][0] ^= 1
        self.assertNotEqual(base, logical_state_hash(mutated_rng))

    def test_malicious_optimizer_metadata_rejected(self) -> None:
        payload, _model, _optimizer, compatibility = self.make_payload()
        bad = copy.deepcopy(payload)
        bad["optimizer_state"]["param_names"].append("evil")
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(bad, run_digest="r" * 64, compatibility=compatibility)
            encoded = encode_decision(observation(), legal_actions())
            model = KernelPolicyValueNet.from_encoded(encoded)
            optimizer = create_adam(model, 0.001)
            load_adam_state(optimizer, model, bad["optimizer_state"], 0.001)


if __name__ == "__main__":
    unittest.main()
