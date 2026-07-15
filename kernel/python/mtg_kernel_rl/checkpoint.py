"""Checkpoint payloads, logical hashes, and Adam state canonicalization."""

from __future__ import annotations

import hashlib
import inspect
import math
import random
import copy
from pathlib import Path
from typing import Any

import torch

from .artifacts import canonical_json_bytes, inject_fault, sha256_bytes, sha256_file
from .determinism import TrainerSeedDerivation, validate_uint63
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
MAX_CHECKPOINT_FILE_BYTES = 64 * 1024 * 1024
MAX_CHECKPOINT_TENSORS = 512
MAX_CHECKPOINT_TENSOR_ELEMENTS = 20_000_000
MAX_CHECKPOINT_TENSOR_BYTES = 256 * 1024 * 1024
MAX_CHECKPOINT_COLLECTION_ITEMS = 4096
MAX_CHECKPOINT_DEPTH = 16
MAX_TORCH_RNG_BYTES = 65536
EXPECTED_TORCH_CPU_RNG_STATE_BYTES = int(torch.random.get_rng_state().numel())
MAX_ADAM_STEP = 1_000_000
ALLOWED_LOGICAL_TENSOR_DTYPES = {
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.complex64,
    torch.complex128,
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


def _require_cpu_strided_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must be a dense strided tensor")
    if getattr(tensor, "is_quantized", False):
        raise ValueError(f"{name} must not be quantized")
    if tensor.numel() > MAX_CHECKPOINT_TENSOR_ELEMENTS:
        raise ValueError(f"{name} tensor has too many elements")
    if tensor.numel() * tensor.element_size() > MAX_CHECKPOINT_TENSOR_BYTES:
        raise ValueError(f"{name} tensor has too many bytes")


def _require_contiguous_cpu_strided_tensor(tensor: torch.Tensor, name: str) -> None:
    _require_cpu_strided_tensor(tensor, name)
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_tensor_values(tensor: torch.Tensor, name: str) -> None:
    _require_cpu_strided_tensor(tensor, name)
    if tensor.dtype not in ALLOWED_LOGICAL_TENSOR_DTYPES:
        raise ValueError(f"{name} has unsupported dtype {tensor.dtype}")
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")


def _clone_validated_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    _validate_tensor_values(tensor, name)
    return tensor.detach().contiguous().clone()


def export_model_state(model: KernelPolicyValueNet) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        out[name] = _clone_validated_tensor(tensor.detach(), f"model_state.{name}")
    return out


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
        _require_cpu_strided_tensor(tensor, f"model_state.{name}")
        if tensor.dtype != expected_tensor.dtype or tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(f"model_state.{name} tensor metadata mismatch")
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"model_state.{name} contains non-finite values")
        out[name] = tensor.detach().contiguous().clone()
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
    nonempty_slots = 0
    step_values: set[int] = set()
    for name, param in named:
        slot = optimizer.state.get(param, {})
        if set(slot) - {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"unexpected Adam slot keys for {name}: {sorted(slot)}")
        if slot and set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"partial Adam slot for {name}")
        out_slot: dict[str, Any] = {}
        for key, value in slot.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Adam slot {name}.{key} must be tensor")
            if key == "step":
                step_values.add(_validate_adam_step_tensor(value, f"optimizer_state.{name}.step"))
                tensor = value.detach().contiguous().clone()
            else:
                tensor = _clone_validated_adam_moment(value, param, f"optimizer_state.{name}.{key}", nonnegative=(key == "exp_avg_sq"))
            out_slot[key] = tensor
        if out_slot:
            nonempty_slots += 1
        state_by_name[name] = out_slot
    if nonempty_slots not in (0, len(named)):
        raise ValueError("Adam state must be empty or complete for every parameter")
    if len(step_values) > 1:
        raise ValueError("Adam step tensors are not synchronized")
    return {
        "schema": "kernel_rl_adam_state/v1",
        "config": settings,
        "param_names": [name for name, _param in named],
        "state": state_by_name,
    }


def _validate_adam_step_tensor(tensor: torch.Tensor, name: str) -> int:
    _require_contiguous_cpu_strided_tensor(tensor, name)
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be torch.float32")
    if tensor.ndim != 0:
        raise ValueError(f"{name} must be rank-0 scalar")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    raw = float(tensor.item())
    if raw < 0.0 or not raw.is_integer():
        raise ValueError(f"{name} must be a finite nonnegative integer")
    value = int(raw)
    if value > MAX_ADAM_STEP:
        raise ValueError(f"{name} exceeds maximum supported Adam step")
    return value


def _clone_validated_adam_moment(tensor: torch.Tensor, param: torch.nn.Parameter, name: str, *, nonnegative: bool) -> torch.Tensor:
    _validate_tensor_values(tensor, name)
    _require_contiguous_cpu_strided_tensor(tensor, name)
    if tensor.dtype != param.dtype or tuple(tensor.shape) != tuple(param.shape) or tensor.device != param.device or tensor.layout != param.layout:
        raise ValueError(f"Adam slot {name} metadata mismatch")
    if nonnegative and torch.any(tensor < 0):
        raise ValueError(f"Adam slot {name} contains negative values")
    return tensor.detach().contiguous().clone()


def _validate_adam_state_metadata(state: Any, *, learning_rate: float | None, optimizer_step_count: int | None) -> None:
    if optimizer_step_count is not None and (
        type(optimizer_step_count) is not int or optimizer_step_count < 0 or optimizer_step_count > MAX_ADAM_STEP
    ):
        raise ValueError("optimizer_step_count out of supported range")
    if not isinstance(state, dict):
        raise ValueError("optimizer_state must be a dict")
    if set(state) != {"schema", "config", "param_names", "state"}:
        raise ValueError("optimizer_state keys mismatch")
    if state["schema"] != "kernel_rl_adam_state/v1":
        raise ValueError("optimizer_state schema mismatch")
    config = state["config"]
    if not isinstance(config, dict) or set(config) != set(adam_config(0.001)):
        raise ValueError("optimizer_state config keys mismatch")
    if config.get("algorithm") != ADAM_ALGORITHM:
        raise ValueError("optimizer_state algorithm mismatch")
    if learning_rate is not None:
        if config != adam_config(learning_rate):
            raise ValueError("optimizer_state config mismatch")
    else:
        lr = config.get("lr")
        if type(lr) is not float or not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("optimizer_state lr must be a positive finite float")
        expected_without_lr = adam_config(lr)
        if config != expected_without_lr:
            raise ValueError("optimizer_state config mismatch")
    param_names = state["param_names"]
    if not isinstance(param_names, list) or not param_names or not all(type(name) is str and name for name in param_names):
        raise ValueError("optimizer_state param_names must be a nonempty string list")
    if len(set(param_names)) != len(param_names):
        raise ValueError("optimizer_state param_names are not unique")
    if len(param_names) > MAX_CHECKPOINT_COLLECTION_ITEMS:
        raise ValueError("optimizer_state has too many parameters")
    state_by_name = state["state"]
    if not isinstance(state_by_name, dict) or set(state_by_name) != set(param_names):
        raise ValueError("optimizer_state parameter keys mismatch")
    nonempty_slots = 0
    for name in param_names:
        raw_slot = state_by_name[name]
        if not isinstance(raw_slot, dict):
            raise ValueError(f"optimizer slot for {name} must be dict")
        if not raw_slot:
            continue
        nonempty_slots += 1
        if set(raw_slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"optimizer slot for {name} must have exact Adam keys")
        step = raw_slot["step"]
        if not isinstance(step, torch.Tensor):
            raise ValueError(f"optimizer slot {name}.step must be tensor")
        step_value = _validate_adam_step_tensor(step, f"optimizer_state.{name}.step")
        if optimizer_step_count is not None and step_value != optimizer_step_count:
            raise ValueError("optimizer step tensor does not match checkpoint counter")
        for key in ("exp_avg", "exp_avg_sq"):
            value = raw_slot[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"optimizer slot {name}.{key} must be tensor")
            _validate_tensor_values(value, f"optimizer_state.{name}.{key}")
            if key == "exp_avg_sq" and torch.any(value < 0):
                raise ValueError(f"optimizer slot {name}.{key} contains negative values")
    if optimizer_step_count == 0 and nonempty_slots:
        raise ValueError("optimizer state must be empty when optimizer_step_count is zero")
    if optimizer_step_count is not None and optimizer_step_count > 0 and nonempty_slots != len(param_names):
        raise ValueError("optimizer state must be complete after optimizer steps")


def load_adam_state(
    optimizer: torch.optim.Adam,
    model: KernelPolicyValueNet,
    state: Any,
    learning_rate: float,
    *,
    expected_step_count: int | None = None,
) -> None:
    _validate_adam_state_metadata(state, learning_rate=learning_rate, optimizer_step_count=expected_step_count)
    named = _named_parameters(model)
    names = [name for name, _param in named]
    if state.get("param_names") != names:
        raise ValueError("optimizer_state param order mismatch")
    state_by_name = state.get("state")
    prepared: list[tuple[torch.nn.Parameter, dict[str, torch.Tensor]]] = []
    for name, param in named:
        raw_slot = state_by_name[name]
        slot: dict[str, torch.Tensor] = {}
        for key, value in raw_slot.items():
            if key == "step":
                _validate_adam_step_tensor(value, f"optimizer_state.{name}.step")
                slot[key] = value.detach().contiguous().clone()
            else:
                slot[key] = _clone_validated_adam_moment(
                    value,
                    param,
                    f"optimizer_state.{name}.{key}",
                    nonnegative=(key == "exp_avg_sq"),
                )
        prepared.append((param, slot))
    optimizer.state.clear()
    for param, slot in prepared:
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
            if key == "step":
                _validate_adam_step_tensor(value, f"optimizer slot {key}")
                continue
            _require_contiguous_cpu_strided_tensor(value, f"optimizer slot {key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"optimizer slot {key} contains non-finite values")
            if key == "exp_avg_sq" and torch.any(value < 0):
                raise ValueError(f"optimizer slot {key} contains negative values")


def capture_python_rng_state() -> dict[str, Any]:
    state = random.getstate()
    return {"version": state[0], "state": list(state[1]), "gauss": state[2]}


def restore_python_rng_state(value: Any) -> None:
    validate_python_rng_state(value)
    random.setstate((value["version"], tuple(value["state"]), value["gauss"]))


def validate_python_rng_state(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"version", "state", "gauss"}:
        raise ValueError("invalid Python RNG state")
    if type(value["version"]) is not int or value["version"] != 3:
        raise ValueError("invalid Python RNG version")
    if (
        not isinstance(value["state"], list)
        or len(value["state"]) != 625
        or not all(type(item) is int and 0 <= item <= 0xFFFF_FFFF for item in value["state"])
    ):
        raise ValueError("invalid Python RNG vector")
    index = value["state"][-1]
    if index < 0 or index > 624:
        raise ValueError("invalid Python RNG index")
    gauss = value["gauss"]
    if gauss is not None and (type(gauss) is not float or not math.isfinite(gauss)):
        raise ValueError("invalid Python RNG gaussian cache")
    isolated = random.Random()
    try:
        isolated.setstate((value["version"], tuple(value["state"]), gauss))
        isolated.random()
    except (TypeError, ValueError) as exc:
        raise ValueError("Python RNG state is not restorable") from exc


def capture_torch_rng_state() -> torch.Tensor:
    return torch.random.get_rng_state().detach().contiguous().clone()


def validate_torch_rng_state(value: Any) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != "cpu":
        raise ValueError("invalid Torch CPU RNG state")
    _require_contiguous_cpu_strided_tensor(value, "Torch CPU RNG state")
    if value.numel() != EXPECTED_TORCH_CPU_RNG_STATE_BYTES or value.numel() > MAX_TORCH_RNG_BYTES:
        raise ValueError("invalid Torch CPU RNG state size")
    if not torch.any(value != 0):
        raise ValueError("Torch CPU RNG state must not be all zero")
    generator = torch.Generator(device="cpu")
    try:
        generator.set_state(value.detach().clone())
        probe = torch.rand(1, generator=generator)
    except RuntimeError as exc:
        raise ValueError("Torch CPU RNG state is not restorable") from exc
    if not torch.isfinite(probe).all():
        raise ValueError("Torch CPU RNG state produced non-finite sample")


def restore_torch_rng_state(value: Any) -> None:
    validate_torch_rng_state(value)
    torch.random.set_rng_state(value.detach().contiguous())


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
        "outcomes_by_learner_seat": copy.deepcopy(outcomes_by_learner_seat),
        "learner_decisions_by_seat": copy.deepcopy(learner_decisions_by_seat),
        "model_config": model.config.to_dict(),
        "model_state": export_model_state(model),
        "optimizer_state": export_adam_state(optimizer, model, learning_rate),
        "python_rng_state": capture_python_rng_state(),
        "torch_cpu_rng_state": capture_torch_rng_state(),
        "base_seed": base_seed,
        "seed_derivation": copy.deepcopy(seed_derivation),
        "provenance": copy.deepcopy(provenance),
        "compatibility": copy.deepcopy(compatibility),
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
    for key in ("completed_update", "optimizer_step_count", "next_episode"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError(f"checkpoint {key} must be a nonnegative int")
    validate_uint63(payload["base_seed"], "checkpoint base_seed")
    if payload["completed_update"] > 0 and payload["next_episode"] <= 0:
        raise ValueError("trained checkpoint must advance next_episode")
    ModelConfig.from_dict(payload["model_config"])
    _validate_counter_map(payload["outcomes_by_learner_seat"], {"win", "loss", "draw"})
    _validate_counter_map(payload["learner_decisions_by_seat"], None)
    _validate_seed_derivation(payload["seed_derivation"])
    _validate_provenance(payload["provenance"])
    _validate_adam_state_metadata(
        payload["optimizer_state"],
        learning_rate=None,
        optimizer_step_count=payload["optimizer_step_count"],
    )
    validate_python_rng_state(payload["python_rng_state"])
    validate_torch_rng_state(payload["torch_cpu_rng_state"])
    return payload


def _validate_seed_derivation(value: Any) -> None:
    expected = TrainerSeedDerivation()
    expected_dict = {
        "version": expected.version,
        "algorithm": expected.algorithm,
        "namespaces": list(expected.namespaces),
    }
    if value != expected_dict:
        raise ValueError("checkpoint seed derivation mismatch")


def _validate_provenance(value: Any) -> None:
    required = {"protocol", "protocol_version", "schema_version", "kernel_version", "surface_version", "card_db_hash"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("checkpoint provenance keys mismatch")
    if value["protocol"] != "kernel_rl_jsonl":
        raise ValueError("checkpoint provenance protocol mismatch")
    for key in ("protocol_version", "schema_version", "surface_version"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"checkpoint provenance {key} must be a nonnegative int")
    if type(value["kernel_version"]) is not str or not value["kernel_version"]:
        raise ValueError("checkpoint provenance kernel_version must be nonempty")
    if type(value["card_db_hash"]) is not int or value["card_db_hash"] < 0 or value["card_db_hash"] > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError("checkpoint provenance card_db_hash out of range")


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
    path = Path(path)
    size = path.stat().st_size
    if size <= 0 or size > MAX_CHECKPOINT_FILE_BYTES:
        raise ValueError("checkpoint file size out of bounds")
    try:
        signature = inspect.signature(torch.load)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Torch safe checkpoint loading is unavailable") from exc
    supports_weights_only = "weights_only" in signature.parameters
    if not supports_weights_only:
        raise RuntimeError("Torch safe checkpoint loading is unavailable")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    _validate_loaded_checkpoint_tree(loaded)
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint root must be a dict")
    return loaded


def save_checkpoint_file(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        torch.save(payload, fh)
        inject_fault("checkpoint_save", path)
        fh.flush()
        inject_fault("checkpoint_flush", path)
        import os

        os.fsync(fh.fileno())
        inject_fault("checkpoint_fsync", path)


def _validate_loaded_checkpoint_tree(value: Any) -> None:
    counters = {"tensors": 0}

    def walk(item: Any, depth: int, context: str) -> None:
        if depth > MAX_CHECKPOINT_DEPTH:
            raise ValueError("checkpoint object nesting is too deep")
        if item is None or type(item) in (bool, int, float, str):
            if type(item) is float and not math.isfinite(item):
                raise ValueError(f"checkpoint non-finite float at {context}")
            return
        if isinstance(item, torch.Tensor):
            counters["tensors"] += 1
            if counters["tensors"] > MAX_CHECKPOINT_TENSORS:
                raise ValueError("checkpoint has too many tensors")
            _validate_tensor_values(item, context)
            return
        if isinstance(item, list):
            if len(item) > MAX_CHECKPOINT_COLLECTION_ITEMS:
                raise ValueError(f"checkpoint list too large at {context}")
            for index, child in enumerate(item):
                walk(child, depth + 1, f"{context}[{index}]")
            return
        if isinstance(item, tuple):
            if len(item) > MAX_CHECKPOINT_COLLECTION_ITEMS:
                raise ValueError(f"checkpoint tuple too large at {context}")
            for index, child in enumerate(item):
                walk(child, depth + 1, f"{context}({index})")
            return
        if isinstance(item, dict):
            if len(item) > MAX_CHECKPOINT_COLLECTION_ITEMS:
                raise ValueError(f"checkpoint dict too large at {context}")
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError(f"checkpoint dict key must be str at {context}")
                walk(child, depth + 1, f"{context}.{key}")
            return
        raise ValueError(f"unsupported checkpoint object type at {context}: {type(item).__name__}")

    walk(value, 0, "$")


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
        _validate_tensor_values(value, "logical_hash.tensor")
        tensor = value.detach().contiguous()
        _hash_atom(hasher, "tensor-dtype", str(tensor.dtype).encode("ascii"))
        shape = b"".join(int(dim).to_bytes(8, "big") for dim in tensor.shape)
        _hash_atom(hasher, "tensor-shape", shape)
        raw = b"" if tensor.numel() == 0 else bytes(tensor.reshape(-1).view(torch.uint8).reshape(-1).tolist())
        _hash_atom(hasher, "tensor-bytes", raw)
    elif isinstance(value, list):
        _hash_atom(hasher, "list-len", len(value).to_bytes(8, "big"))
        for item in value:
            _logical_update(hasher, item)
    elif isinstance(value, tuple):
        _hash_atom(hasher, "tuple-len", len(value).to_bytes(8, "big"))
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
