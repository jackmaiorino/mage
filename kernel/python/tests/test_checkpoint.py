from __future__ import annotations

import copy
import dataclasses
import random
import tempfile
import unittest
from pathlib import Path

import torch

from mtg_kernel_rl.checkpoint import (
    MAX_CHECKPOINT_COLLECTION_ITEMS,
    MAX_CHECKPOINT_TENSOR_ELEMENTS,
    assert_model_finite,
    assert_optimizer_finite,
    build_checkpoint_payload,
    create_adam,
    export_adam_state,
    load_adam_state,
    load_checkpoint_file,
    logical_state_hash,
    save_checkpoint_file,
    validate_checkpoint_payload,
    validate_model_state,
    validate_python_rng_state,
    validate_torch_rng_state,
)
import mtg_kernel_rl.checkpoint as checkpoint_mod
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
            seed_derivation={
                **dataclasses.asdict(TrainerSeedDerivation()),
                "namespaces": list(TrainerSeedDerivation().namespaces),
            },
            provenance={"protocol": "kernel_rl_jsonl", "protocol_version": 2, "schema_version": 2, "kernel_version": "0.0.1-spike", "surface_version": 2, "card_db_hash": 1},
            compatibility=compatibility,
        )
        return payload, model, optimizer, compatibility

    def assert_payload_equal(self, left: object, right: object, context: str = "$") -> None:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            self.assertIsInstance(left, torch.Tensor, context)
            self.assertIsInstance(right, torch.Tensor, context)
            self.assertTrue(torch.equal(left, right), context)
            return
        if isinstance(left, dict) or isinstance(right, dict):
            self.assertIsInstance(left, dict, context)
            self.assertIsInstance(right, dict, context)
            self.assertEqual(set(left), set(right), context)
            for key in left:
                self.assert_payload_equal(left[key], right[key], f"{context}.{key}")
            return
        if isinstance(left, list) or isinstance(right, list):
            self.assertIsInstance(left, list, context)
            self.assertIsInstance(right, list, context)
            self.assertEqual(len(left), len(right), context)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assert_payload_equal(a, b, f"{context}[{index}]")
            return
        self.assertEqual(left, right, context)

    def assert_optimizer_payload_rejected(self, payload: dict, model: KernelPolicyValueNet, compatibility: dict) -> None:
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(payload, run_digest="r" * 64, compatibility=compatibility)
        restored = KernelPolicyValueNet(model.config, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=0)
        optimizer = create_adam(restored, 0.001)
        with self.assertRaises(ValueError):
            load_adam_state(
                optimizer,
                restored,
                payload["optimizer_state"],
                0.001,
                expected_step_count=payload["optimizer_step_count"],
            )

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
        self.assertNotEqual(logical_state_hash({"x": [1, 2]}), logical_state_hash({"x": (1, 2)}))

    def test_malicious_optimizer_metadata_rejected(self) -> None:
        payload, _model, _optimizer, compatibility = self.make_payload()
        bad = copy.deepcopy(payload)
        bad["optimizer_state"]["param_names"].append("evil")
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(bad, run_digest="r" * 64, compatibility=compatibility)

    def test_malformed_adam_slots_are_rejected(self) -> None:
        payload, model, _optimizer, compatibility = self.make_payload()
        first_name = payload["optimizer_state"]["param_names"][0]
        bad_partial = copy.deepcopy(payload)
        del bad_partial["optimizer_state"]["state"][first_name]["exp_avg_sq"]
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(bad_partial, run_digest="r" * 64, compatibility=compatibility)
        bad_amsgrad = copy.deepcopy(payload)
        bad_amsgrad["optimizer_state"]["state"][first_name]["max_exp_avg_sq"] = torch.zeros_like(
            bad_amsgrad["optimizer_state"]["state"][first_name]["exp_avg"]
        )
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(bad_amsgrad, run_digest="r" * 64, compatibility=compatibility)
        bad_step = copy.deepcopy(payload)
        bad_step["optimizer_state"]["state"][first_name]["step"] += 1
        restored = KernelPolicyValueNet(model.config, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=0)
        optimizer = create_adam(restored, 0.001)
        with self.assertRaises(ValueError):
            load_adam_state(
                optimizer,
                restored,
                bad_step["optimizer_state"],
                0.001,
                expected_step_count=payload["optimizer_step_count"],
            )

    def test_adam_step_tensor_adversarial_forms_are_rejected_before_install(self) -> None:
        payload, model, _optimizer, compatibility = self.make_payload()
        first_name = payload["optimizer_state"]["param_names"][0]
        cases = {
            "rank1_singleton": torch.tensor([1.0], dtype=torch.float32),
            "bool": torch.tensor(True),
            "integer": torch.tensor(1, dtype=torch.int64),
            "float64": torch.tensor(1.0, dtype=torch.float64),
            "fractional": torch.tensor(1.5, dtype=torch.float32),
            "negative": torch.tensor(-1.0, dtype=torch.float32),
            "nonfinite": torch.tensor(float("inf"), dtype=torch.float32),
            "over_bound": torch.tensor(float(checkpoint_mod.MAX_ADAM_STEP + 1), dtype=torch.float32),
        }
        for name, step in cases.items():
            with self.subTest(name=name):
                bad = copy.deepcopy(payload)
                bad["optimizer_state"]["state"][first_name]["step"] = step
                self.assert_optimizer_payload_rejected(bad, model, compatibility)

    def test_adam_moment_metadata_and_negative_second_moment_are_rejected(self) -> None:
        payload, model, _optimizer, compatibility = self.make_payload()
        first_name = payload["optimizer_state"]["param_names"][0]
        negative = copy.deepcopy(payload)
        negative["optimizer_state"]["state"][first_name]["exp_avg_sq"] = negative["optimizer_state"]["state"][first_name]["exp_avg_sq"].clone()
        negative["optimizer_state"]["state"][first_name]["exp_avg_sq"].reshape(-1)[0] = -1.0
        self.assert_optimizer_payload_rejected(negative, model, compatibility)

        load_only_cases = {}
        shape_bad = copy.deepcopy(payload)
        shape_bad["optimizer_state"]["state"][first_name]["exp_avg"] = torch.zeros(1, dtype=torch.float32)
        load_only_cases["shape"] = shape_bad
        dtype_bad = copy.deepcopy(payload)
        dtype_bad["optimizer_state"]["state"][first_name]["exp_avg"] = dtype_bad["optimizer_state"]["state"][first_name]["exp_avg"].double()
        load_only_cases["dtype"] = dtype_bad
        contiguous_bad = copy.deepcopy(payload)
        source = contiguous_bad["optimizer_state"]["state"][first_name]["exp_avg"]
        expanded = torch.zeros(tuple(dim * 2 for dim in source.shape), dtype=source.dtype)
        slices = tuple(slice(None, None, 2) for _ in source.shape)
        contiguous_bad["optimizer_state"]["state"][first_name]["exp_avg"] = expanded[slices]
        self.assertFalse(contiguous_bad["optimizer_state"]["state"][first_name]["exp_avg"].is_contiguous())
        load_only_cases["noncontiguous"] = contiguous_bad
        for name, bad in load_only_cases.items():
            with self.subTest(name=name):
                validate_checkpoint_payload(bad, run_digest="r" * 64, compatibility=compatibility)
                restored = KernelPolicyValueNet(model.config, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=0)
                optimizer = create_adam(restored, 0.001)
                with self.assertRaises(ValueError):
                    load_adam_state(
                        optimizer,
                        restored,
                        bad["optimizer_state"],
                        0.001,
                        expected_step_count=bad["optimizer_step_count"],
                    )

    def test_valid_loaded_adam_state_can_step_without_mutating_source_payload(self) -> None:
        payload, model, _optimizer, compatibility = self.make_payload()
        validate_checkpoint_payload(payload, run_digest="r" * 64, compatibility=compatibility)
        source_copy = copy.deepcopy(payload)
        restored = KernelPolicyValueNet(model.config, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=0)
        restored.load_state_dict(validate_model_state(restored, payload["model_state"]), strict=True)
        optimizer = create_adam(restored, 0.001)
        load_adam_state(optimizer, restored, payload["optimizer_state"], 0.001, expected_step_count=payload["optimizer_step_count"])
        encoded = encode_decision(observation(), legal_actions())
        logits, value = restored(encoded)
        loss = logits.square().mean() + value.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert_model_finite(restored)
        assert_optimizer_finite(optimizer)
        export_adam_state(optimizer, restored, 0.001)
        self.assert_payload_equal(payload, source_copy)

    def test_rng_validators_reject_unrestorable_states_and_do_not_touch_globals(self) -> None:
        payload, _model, _optimizer, compatibility = self.make_payload()
        py_global = random.getstate()
        torch_global = torch.random.get_rng_state().clone()
        validate_python_rng_state(payload["python_rng_state"])
        validate_torch_rng_state(payload["torch_cpu_rng_state"])
        bad_py = copy.deepcopy(payload["python_rng_state"])
        bad_py["state"][-1] = 625
        with self.assertRaises(ValueError):
            validate_python_rng_state(bad_py)
        bad_torch = torch.zeros_like(payload["torch_cpu_rng_state"])
        with self.assertRaises(ValueError):
            validate_torch_rng_state(bad_torch)
        checkpoint_bad_py = copy.deepcopy(payload)
        checkpoint_bad_py["python_rng_state"] = bad_py
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(checkpoint_bad_py, run_digest="r" * 64, compatibility=compatibility)
        checkpoint_bad_torch = copy.deepcopy(payload)
        checkpoint_bad_torch["torch_cpu_rng_state"] = bad_torch
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(checkpoint_bad_torch, run_digest="r" * 64, compatibility=compatibility)
        self.assertEqual(py_global, random.getstate())
        self.assertTrue(torch.equal(torch_global, torch.random.get_rng_state()))

    def test_safe_load_incompatibility_never_calls_unsafe_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "bad.pt"
            path.write_bytes(b"not a checkpoint")
            called = {"value": False}
            original = checkpoint_mod.torch.load

            def unsafe_load(path, map_location=None):  # type: ignore[no-untyped-def]
                called["value"] = True
                return {}

            checkpoint_mod.torch.load = unsafe_load  # type: ignore[assignment]
            try:
                with self.assertRaises(RuntimeError):
                    load_checkpoint_file(path)
            finally:
                checkpoint_mod.torch.load = original  # type: ignore[assignment]
            self.assertFalse(called["value"])

    def test_loaded_checkpoint_tree_bounds_reject_oversized_tensors_and_collections(self) -> None:
        original_elements = checkpoint_mod.MAX_CHECKPOINT_TENSOR_ELEMENTS
        original_items = checkpoint_mod.MAX_CHECKPOINT_COLLECTION_ITEMS
        try:
            checkpoint_mod.MAX_CHECKPOINT_TENSOR_ELEMENTS = 1
            with self.assertRaises(ValueError):
                logical_state_hash({"x": torch.zeros(2, dtype=torch.float32)})
            checkpoint_mod.MAX_CHECKPOINT_COLLECTION_ITEMS = 1
            with tempfile.TemporaryDirectory() as tmp_name:
                path = Path(tmp_name) / "oversized.pt"
                torch.save({"a": 1, "b": 2}, path)
                with self.assertRaises(ValueError):
                    load_checkpoint_file(path)
        finally:
            checkpoint_mod.MAX_CHECKPOINT_TENSOR_ELEMENTS = original_elements
            checkpoint_mod.MAX_CHECKPOINT_COLLECTION_ITEMS = original_items


if __name__ == "__main__":
    unittest.main()
