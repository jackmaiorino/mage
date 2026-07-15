"""Checkpoint payloads, logical hashes, and Adam state canonicalization."""

from __future__ import annotations

import hashlib
import math
import random
import ctypes
from pathlib import Path
from typing import Any

import torch

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from .model import KernelPolicyValueNet, ModelConfig

CHECKPOINT_SCHEMA = "kernel_rl_train_checkpoint/v1"
SIDECAR_SCHEMA = "kernel_rl_train_checkpoint_sidecar/v1"
UPDATE_RECORD_SCHEMA = "kernel_rl_train_update_record/v1"
LATEST_SCHEMA = "kernel_rl_train_latest/v1"
ADAM_ALGORITHM = "adam/torch-cpu-canonical-v1"
ADAM_SETTINGS = {
    "lr": None,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "foreach": False,
    "fused": False,
    "maximize": False,
    "capturable": False,
    "differentiable": False,
}


def create_adam(model: KernelPolicyValueNet, learning_rate: float) -> torch.optim.Adam:
    if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a positive finite float")
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        fused=False,
    )


def adam_config(learning_rate: float) -> dict[str, Any]:
    config = dict(ADAM_SETTINGS)
    config["lr"] = learning_rate
    config["algorithm"] = ADAM_ALGORITHM
    return config


def _named_parameters(model: KernelPolicyValueNet) -> list[tuple[str, torch.nn.Parameter]]:
    return list(model.named_parameters())


def export_model_state(model: KernelPolicyValueNet) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.state_dict().items()}


def validate_model_state(model: KernelPolicyValueNet, state: Any) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise ValueError("model_state must be a dict")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("model_state keys mismatch")
    out: dict[str, torch.Tensor] = {}
    for name, expected_tensor in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"model_state.{name} must be a tensor")
        if tensor.dtype != expected_tensor.dtype or tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(f"model_state.{name} tensor metadata mismatch")
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"model_state.{name} contains non-finite values")
        out[name] = tensor.detach().cpu().contiguous().clone()
    return out


def assert_model_finite(model: KernelPolicyValueNet) -> None:
    for name, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"model state {name} contains non-finite values")


def export_adam_state(optimizer: torch.optim.Adam, model: KernelPolicyValueNet, learning_rate: float) -> dict[str, Any]:
    named = _named_parameters(model)
    group = optimizer.param_groups
    if len(group) != 1:
        raise ValueError("expected one Adam parameter group")
    params = list(group[0]["params"])
    expected_params = [param for _name, param in named]
    if params != expected_params:
        raise ValueError("Adam parameter order drifted")
    settings = adam_config(learning_rate)
    for key, expected in settings.items():
        if key == "algorithm":
            continue
        actual = group[0].get(key)
        if key == "betas":
            actual = list(actual)
        if actual != expected:
            raise ValueError(f"Adam setting {key} drifted: {actual!r} != {expected!r}")
    state_by_name: dict[str, dict[str, Any]] = {}
    for name, param in named:
        slot = optimizer.state.get(param, {})
        if set(slot) - {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
            raise ValueError(f"unexpected Adam slot keys for {name}: {sorted(slot)}")
        out_slot: dict[str, Any] = {}
        for key, value in slot.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Adam slot {name}.{key} must be tensor")
            tensor = value.detach().cpu().contiguous().clone()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Adam slot {name}.{key} contains non-finite values")
            if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} and tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(f"Adam slot {name}.{key} shape mismatch")
            if key == "step" and tensor.numel() != 1:
                raise ValueError(f"Adam slot {name}.step must be scalar")
            out_slot[key] = tensor
        state_by_name[name] = out_slot
    return {
        "schema": "kernel_rl_adam_state/v1",
        "config": settings,
        "param_names": [name for name, _param in named],
        "state": state_by_name,
    }


def load_adam_state(optimizer: torch.optim.Adam, model: KernelPolicyValueNet, state: Any, learning_rate: float) -> None:
    if not isinstance(state, dict):
        raise ValueError("optimizer_state must be a dict")
    if state.get("schema") != "kernel_rl_adam_state/v1":
        raise ValueError("optimizer_state schema mismatch")
    expected_config = adam_config(learning_rate)
    if state.get("config") != expected_config:
        raise ValueError("optimizer_state config mismatch")
    named = _named_parameters(model)
    names = [name for name, _param in named]
    if state.get("param_names") != names:
        raise ValueError("optimizer_state param order mismatch")
    state_by_name = state.get("state")
    if not isinstance(state_by_name, dict) or set(state_by_name) != set(names):
        raise ValueError("optimizer_state parameter keys mismatch")
    optimizer.state.clear()
    for name, param in named:
        raw_slot = state_by_name[name]
        if not isinstance(raw_slot, dict):
            raise ValueError(f"optimizer slot for {name} must be dict")
        if set(raw_slot) - {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
            raise ValueError(f"unexpected optimizer slot keys for {name}")
        slot: dict[str, torch.Tensor] = {}
        for key, value in raw_slot.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"optimizer slot {name}.{key} must be tensor")
            tensor = value.detach().cpu().contiguous().clone()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"optimizer slot {name}.{key} contains non-finite values")
            if key == "step":
                if tensor.numel() != 1:
                    raise ValueError(f"optimizer slot {name}.step must be scalar")
            elif tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(f"optimizer slot {name}.{key} shape mismatch")
            slot[key] = tensor
        if slot:
            optimizer.state[param] = slot
    export_adam_state(optimizer, model, learning_rate)


def assert_optimizer_finite(optimizer: torch.optim.Adam) -> None:
    for param, slot in optimizer.state.items():
        if not isinstance(param, torch.nn.Parameter):
            raise ValueError("optimizer has non-parameter state key")
        for key, value in slot.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"optimizer slot {key} must be tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"optimizer slot {key} contains non-finite values")


def capture_python_rng_state() -> dict[str, Any]:
    state = random.getstate()
    return {"version": state[0], "state": list(state[1]), "gauss": state[2]}


def restore_python_rng_state(value: Any) -> None:
    validate_python_rng_state(value)
    random.setstate((value["version"], tuple(value["state"]), value["gauss"]))


def validate_python_rng_state(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"version", "state", "gauss"}:
        raise ValueError("invalid Python RNG state")
    if type(value["version"]) is not int:
        raise ValueError("invalid Python RNG version")
    if not isinstance(value["state"], list) or not all(type(item) is int for item in value["state"]):
        raise ValueError("invalid Python RNG vector")
    gauss = value["gauss"]
    if gauss is not None and (type(gauss) is not float or not math.isfinite(gauss)):
        raise ValueError("invalid Python RNG gaussian cache")


def capture_torch_rng_state() -> torch.Tensor:
    return torch.random.get_rng_state().detach().cpu().contiguous().clone()


def restore_torch_rng_state(value: Any) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError("invalid Torch CPU RNG state")
    torch.random.set_rng_state(value.detach().cpu().contiguous())


def build_checkpoint_payload(
    *,
    run_digest: str,
    completed_update: int,
    optimizer_step_count: int,
    next_episode: int,
    outcomes_by_learner_seat: dict[str, dict[str, int]],
    learner_decisions_by_seat: dict[str, int],
    model: KernelPolicyValueNet,
    optimizer: torch.optim.Adam,
    learning_rate: float,
    base_seed: int,
    seed_derivation: dict[str, Any],
    provenance: dict[str, Any],
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "run_digest": run_digest,
        "completed_update": completed_update,
        "optimizer_step_count": optimizer_step_count,
        "next_episode": next_episode,
        "outcomes_by_learner_seat": outcomes_by_learner_seat,
        "learner_decisions_by_seat": learner_decisions_by_seat,
        "model_config": model.config.to_dict(),
        "model_state": export_model_state(model),
        "optimizer_state": export_adam_state(optimizer, model, learning_rate),
        "python_rng_state": capture_python_rng_state(),
        "torch_cpu_rng_state": capture_torch_rng_state(),
        "base_seed": base_seed,
        "seed_derivation": seed_derivation,
        "provenance": provenance,
        "compatibility": compatibility,
    }


def validate_checkpoint_payload(payload: Any, *, run_digest: str, compatibility: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")
    required = {
        "schema",
        "run_digest",
        "completed_update",
        "optimizer_step_count",
        "next_episode",
        "outcomes_by_learner_seat",
        "learner_decisions_by_seat",
        "model_config",
        "model_state",
        "optimizer_state",
        "python_rng_state",
        "torch_cpu_rng_state",
        "base_seed",
        "seed_derivation",
        "provenance",
        "compatibility",
    }
    if set(payload) != required:
        raise ValueError("checkpoint payload keys mismatch")
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    if payload["run_digest"] != run_digest:
        raise ValueError("checkpoint run_digest mismatch")
    if payload["compatibility"] != compatibility:
        raise ValueError("checkpoint compatibility mismatch")
    for key in ("completed_update", "optimizer_step_count", "next_episode", "base_seed"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError(f"checkpoint {key} must be a nonnegative int")
    if payload["completed_update"] > 0 and payload["next_episode"] <= 0:
        raise ValueError("trained checkpoint must advance next_episode")
    ModelConfig.from_dict(payload["model_config"])
    _validate_counter_map(payload["outcomes_by_learner_seat"], {"win", "loss", "draw"})
    _validate_counter_map(payload["learner_decisions_by_seat"], None)
    validate_python_rng_state(payload["python_rng_state"])
    if not isinstance(payload["torch_cpu_rng_state"], torch.Tensor) or payload["torch_cpu_rng_state"].dtype != torch.uint8:
        raise ValueError("invalid Torch RNG state in checkpoint")
    return payload


def _validate_counter_map(value: Any, nested_keys: set[str] | None) -> None:
    if not isinstance(value, dict) or set(value) != {"p0", "p1"}:
        raise ValueError("counter map must have p0/p1 keys")
    if nested_keys is None:
        for seat in ("p0", "p1"):
            if type(value[seat]) is not int or value[seat] < 0:
                raise ValueError("seat counter must be a nonnegative int")
        return
    for seat in ("p0", "p1"):
        raw = value[seat]
        if not isinstance(raw, dict) or set(raw) != nested_keys:
            raise ValueError("nested seat counter keys mismatch")
        for count in raw.values():
            if type(count) is not int or count < 0:
                raise ValueError("nested seat counter must be a nonnegative int")


def load_checkpoint_file(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_checkpoint_file(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        import os

        os.fsync(fh.fileno())


def _hash_atom(hasher: Any, tag: str, payload: bytes) -> None:
    tag_bytes = tag.encode("utf-8")
    hasher.update(len(tag_bytes).to_bytes(4, "big"))
    hasher.update(tag_bytes)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _logical_update(hasher: Any, value: Any) -> None:
    if value is None:
        _hash_atom(hasher, "none", b"")
    elif type(value) is bool:
        _hash_atom(hasher, "bool", b"\x01" if value else b"\x00")
    elif type(value) is int:
        if value < 0:
            _hash_atom(hasher, "int-neg", str(value).encode("ascii"))
        else:
            _hash_atom(hasher, "int", value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big"))
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("logical hash rejects non-finite float")
        _hash_atom(hasher, "float", value.hex().encode("ascii"))
    elif type(value) is str:
        _hash_atom(hasher, "str", value.encode("utf-8"))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError("logical hash rejects non-finite tensor")
        _hash_atom(hasher, "tensor-dtype", str(tensor.dtype).encode("ascii"))
        shape = b"".join(int(dim).to_bytes(8, "big") for dim in tensor.shape)
        _hash_atom(hasher, "tensor-shape", shape)
        raw = b"" if tensor.numel() == 0 else ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())
        _hash_atom(hasher, "tensor-bytes", raw)
    elif isinstance(value, (list, tuple)):
        _hash_atom(hasher, "list-len", len(value).to_bytes(8, "big"))
        for item in value:
            _logical_update(hasher, item)
    elif isinstance(value, dict):
        for key in value:
            if type(key) is not str:
                raise TypeError("logical hash only supports string dict keys")
        _hash_atom(hasher, "dict-len", len(value).to_bytes(8, "big"))
        for key in sorted(value):
            _logical_update(hasher, key)
            _logical_update(hasher, value[key])
    else:
        raise TypeError(f"unsupported logical hash type: {type(value).__name__}")


def logical_state_hash(payload: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    _logical_update(hasher, payload)
    return hasher.hexdigest()


def update_record_hash(record: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(record))


def compute_head(
    *,
    parent_head: str | None,
    checkpoint_byte_hash: str,
    logical_hash: str,
    update_hash: str,
) -> str:
    payload = {
        "schema": "kernel_rl_train_head/v1",
        "parent_head": parent_head,
        "checkpoint_byte_hash": checkpoint_byte_hash,
        "logical_state_hash": logical_hash,
        "update_record_hash": update_hash,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def build_sidecar(
    *,
    update: int,
    run_digest: str,
    parent_head: str | None,
    checkpoint_sha256: str,
    logical_hash: str,
    update_hash: str,
) -> dict[str, Any]:
    head = compute_head(
        parent_head=parent_head,
        checkpoint_byte_hash=checkpoint_sha256,
        logical_hash=logical_hash,
        update_hash=update_hash,
    )
    return {
        "schema": SIDECAR_SCHEMA,
        "update": update,
        "run_digest": run_digest,
        "parent_head": parent_head,
        "checkpoint_sha256": checkpoint_sha256,
        "logical_state_sha256": logical_hash,
        "update_record_sha256": update_hash,
        "head": head,
    }


def build_latest(*, update: int, run_digest: str, head: str) -> dict[str, Any]:
    return {"schema": LATEST_SCHEMA, "update": update, "run_digest": run_digest, "head": head}
