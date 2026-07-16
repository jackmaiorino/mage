"""Small deterministic CPU Torch model for variable legal actions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .determinism import configure_torch_determinism
from .features import (
    ACTION_FEATURE_DIM,
    ACTION_REF_FEATURE_DIM,
    CARD_TOKEN_VOCAB_SIZE,
    EDGE_FEATURE_DIM,
    EncodedDecision,
    FEATURE_REGISTRY_VERSION,
    FEATURE_SCHEMA_VERSION,
    OBJECT_FEATURE_DIM,
    OBJECT_GROUPS,
    STATE_FEATURE_DIM,
    encoding_contract_fingerprint,
    feature_contract_fingerprint,
)

MODEL_CONFIG_SCHEMA_VERSION = 4
MODEL_ARCHITECTURE_VERSION = "kernel-policy-value-net-5"
MODEL_CARD_EMBEDDING_DIM = 16
MODEL_HIDDEN_DIM = 64
INITIALIZER_RUNNER_FIXED_V1 = "runner-fixed-v1"
INITIALIZER_TRAINER_SEEDED_V1 = "trainer-seeded-v1"


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int = MODEL_CONFIG_SCHEMA_VERSION
    model_architecture_version: str = MODEL_ARCHITECTURE_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_registry_version: str = FEATURE_REGISTRY_VERSION
    feature_contract_digest: str = feature_contract_fingerprint()
    feature_encoding_digest: str = encoding_contract_fingerprint()
    card_vocab_size: int = CARD_TOKEN_VOCAB_SIZE
    card_embedding_dim: int = MODEL_CARD_EMBEDDING_DIM
    hidden_dim: int = MODEL_HIDDEN_DIM
    state_dim: int = STATE_FEATURE_DIM
    object_feature_dim: int = OBJECT_FEATURE_DIM
    edge_feature_dim: int = EDGE_FEATURE_DIM
    action_feature_dim: int = ACTION_FEATURE_DIM
    object_group_count: int = len(OBJECT_GROUPS)
    action_ref_feature_dim: int = ACTION_REF_FEATURE_DIM

    def to_dict(self) -> dict[str, int | str]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        if not isinstance(value, dict):
            raise TypeError("ModelConfig input must be a primitive dictionary")
        fields = cls.__dataclass_fields__
        expected = set(fields)
        actual = set(value)
        if expected != actual:
            raise ValueError(f"ModelConfig fields mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
        int_fields = {
            "schema_version",
            "card_vocab_size",
            "card_embedding_dim",
            "hidden_dim",
            "state_dim",
            "object_feature_dim",
            "edge_feature_dim",
            "action_feature_dim",
            "object_group_count",
            "action_ref_feature_dim",
        }
        str_fields = set(fields) - int_fields
        kwargs: dict[str, Any] = {}
        for key in fields:
            raw = value[key]
            if key in int_fields:
                if type(raw) is not int:
                    raise TypeError(f"ModelConfig.{key} must be int")
            elif key in str_fields:
                if type(raw) is not str:
                    raise TypeError(f"ModelConfig.{key} must be str")
            else:
                raise TypeError(f"unsupported ModelConfig field {key}")
            kwargs[key] = raw
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != MODEL_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported ModelConfig schema_version")
        if self.model_architecture_version != MODEL_ARCHITECTURE_VERSION:
            raise ValueError("unsupported model architecture version")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("feature schema version mismatch")
        if self.feature_registry_version != FEATURE_REGISTRY_VERSION:
            raise ValueError("feature registry version mismatch")
        if self.feature_contract_digest != feature_contract_fingerprint():
            raise ValueError("feature contract digest mismatch")
        if self.feature_encoding_digest != encoding_contract_fingerprint():
            raise ValueError("feature encoding digest mismatch")
        exact_ints = {
            "card_vocab_size": (self.card_vocab_size, CARD_TOKEN_VOCAB_SIZE),
            "card_embedding_dim": (self.card_embedding_dim, MODEL_CARD_EMBEDDING_DIM),
            "hidden_dim": (self.hidden_dim, MODEL_HIDDEN_DIM),
            "state_dim": (self.state_dim, STATE_FEATURE_DIM),
            "object_feature_dim": (self.object_feature_dim, OBJECT_FEATURE_DIM),
            "edge_feature_dim": (self.edge_feature_dim, EDGE_FEATURE_DIM),
            "action_feature_dim": (self.action_feature_dim, ACTION_FEATURE_DIM),
            "object_group_count": (self.object_group_count, len(OBJECT_GROUPS)),
            "action_ref_feature_dim": (self.action_ref_feature_dim, ACTION_REF_FEATURE_DIM),
        }
        for key, (raw, expected) in exact_ints.items():
            if type(raw) is not int or raw != expected:
                raise ValueError(f"ModelConfig.{key} must equal contract value {expected}")

    def contract_fingerprint(self) -> str:
        payload = self.to_dict()
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class KernelPolicyValueNet(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        initializer: str = INITIALIZER_RUNNER_FIXED_V1,
        initializer_seed: int | None = None,
        configure_runtime: bool = True,
    ) -> None:
        super().__init__()
        if configure_runtime:
            configure_torch_determinism()
        config.validate()
        self.config = config
        factory = {"device": "meta", "dtype": torch.float32}
        self.card_embedding = nn.Embedding(config.card_vocab_size, config.card_embedding_dim, padding_idx=0, **factory)
        self.object_encoder = nn.Sequential(
            nn.Linear(config.object_feature_dim + config.card_embedding_dim, config.hidden_dim, **factory),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim, **factory),
            nn.Tanh(),
        )
        pooled_dim = config.hidden_dim * config.object_group_count
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.edge_feature_dim + config.hidden_dim * 2, config.hidden_dim, **factory),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim, **factory),
            nn.Tanh(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim, **factory),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim, **factory),
            nn.Tanh(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim + pooled_dim, config.hidden_dim, **factory),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim, **factory),
            nn.Tanh(),
        )
        self.action_ref_encoder = nn.Sequential(
            nn.Linear(config.action_ref_feature_dim + config.hidden_dim, config.hidden_dim, **factory),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim, **factory),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_feature_dim + config.hidden_dim, config.hidden_dim, **factory),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim, **factory),
            nn.Tanh(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim, **factory),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1, **factory),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim, **factory),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1, **factory),
        )
        self.to_empty(device="cpu")
        if initializer == INITIALIZER_RUNNER_FIXED_V1:
            self.reset_deterministic_parameters()
        elif initializer == INITIALIZER_TRAINER_SEEDED_V1:
            if initializer_seed is None:
                raise ValueError("trainer seeded initializer requires initializer_seed")
            self.reset_seeded_parameters(initializer_seed)
        else:
            raise ValueError(f"unsupported model initializer {initializer}")

    @classmethod
    def from_encoded(
        cls,
        encoded: EncodedDecision,
        *,
        card_vocab_size: int = CARD_TOKEN_VOCAB_SIZE,
        card_embedding_dim: int = MODEL_CARD_EMBEDDING_DIM,
        hidden_dim: int = MODEL_HIDDEN_DIM,
    ) -> "KernelPolicyValueNet":
        cfg = model_config_from_encoded(
            encoded,
            card_vocab_size=card_vocab_size,
            card_embedding_dim=card_embedding_dim,
            hidden_dim=hidden_dim,
        )
        return cls(cfg)

    def reset_deterministic_parameters(self) -> None:
        with torch.no_grad():
            for _name, param in self.named_parameters():
                if param.ndim == 1:
                    values = torch.linspace(-0.05, 0.05, param.numel(), dtype=param.dtype, device=param.device)
                else:
                    values = torch.arange(param.numel(), dtype=param.dtype, device=param.device)
                    values = ((values % 31) - 15) / 200.0
                param.copy_(values.reshape_as(param))
            self.card_embedding.weight[0].zero_()

    def reset_seeded_parameters(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("initializer seed must be a nonnegative integer and not bool")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        with torch.no_grad():
            for _name, param in self.named_parameters():
                if param.ndim >= 2:
                    values = torch.empty(param.shape, dtype=param.dtype, device=param.device)
                    torch.nn.init.xavier_uniform_(values, generator=generator)
                else:
                    values = torch.rand(param.shape, dtype=param.dtype, device=param.device, generator=generator) * 0.02 - 0.01
                param.copy_(values)
            self.card_embedding.weight[0].zero_()

    def forward(self, encoded: EncodedDecision) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_encoded(encoded)
        object_card = self.card_embedding(encoded.object_card_ids)
        object_input = torch.cat([encoded.object_features, object_card], dim=-1)
        object_base_hidden = self.object_encoder(object_input)
        edge_pooled = torch.zeros_like(object_base_hidden)
        if encoded.edge_features.shape[0] > 0:
            edge_input = torch.cat(
                [
                    encoded.edge_features,
                    object_base_hidden[encoded.edge_source_indices],
                    object_base_hidden[encoded.edge_target_indices],
                ],
                dim=-1,
            )
            edge_hidden = self.edge_encoder(edge_input)
            edge_pooled.index_add_(0, encoded.edge_source_indices, edge_hidden)
            edge_pooled.index_add_(0, encoded.edge_target_indices, edge_hidden)
        object_hidden = self.node_update(torch.cat([object_base_hidden, edge_pooled], dim=-1))
        pooled = torch.zeros(
            self.config.object_group_count,
            self.config.hidden_dim,
            dtype=object_hidden.dtype,
            device=object_hidden.device,
        )
        pooled.index_add_(0, encoded.object_groups, object_hidden)
        pooled_flat = pooled.reshape(-1)
        state_input = torch.cat([encoded.state, pooled_flat], dim=0)
        state_hidden = self.state_encoder(state_input)

        action_count = encoded.action_features.shape[0]
        action_ref_pooled = torch.zeros(
            action_count,
            self.config.hidden_dim,
            dtype=encoded.action_features.dtype,
            device=encoded.action_features.device,
        )
        if encoded.action_ref_features.shape[0] > 0:
            action_ref_nodes = object_hidden[encoded.action_ref_node_indices]
            action_ref_input = torch.cat([encoded.action_ref_features, action_ref_nodes], dim=-1)
            action_ref_hidden = _apply_rowwise(self.action_ref_encoder, action_ref_input)
            action_ref_pooled.index_add_(0, encoded.action_ref_action_indices, action_ref_hidden)
        action_input = torch.cat([encoded.action_features, action_ref_pooled], dim=-1)
        action_hidden = _apply_rowwise(self.action_encoder, action_input)
        repeated_state = state_hidden.unsqueeze(0).expand(action_hidden.shape[0], -1)
        logits = _apply_rowwise(self.scorer, torch.cat([repeated_state, action_hidden], dim=-1)).squeeze(-1)
        value = self.value_head(state_hidden).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("model produced non-finite logits")
        if not torch.isfinite(value).all():
            raise ValueError("model produced non-finite value")
        return logits, value

    def _validate_encoded(self, encoded: EncodedDecision) -> None:
        schema = encoded.schema
        if schema.version != self.config.feature_schema_version:
            raise ValueError("encoded feature schema version mismatch")
        if schema.registry_version != self.config.feature_registry_version:
            raise ValueError("encoded feature registry version mismatch")
        if schema.contract_digest != self.config.feature_contract_digest:
            raise ValueError("encoded feature contract digest mismatch")
        if schema.encoding_digest != self.config.feature_encoding_digest:
            raise ValueError("encoded feature encoding digest mismatch")
        if schema.state_dim != self.config.state_dim:
            raise ValueError("encoded state_dim mismatch")
        if schema.object_feature_dim != self.config.object_feature_dim:
            raise ValueError("encoded object_feature_dim mismatch")
        if schema.edge_feature_dim != self.config.edge_feature_dim:
            raise ValueError("encoded edge_feature_dim mismatch")
        if schema.action_feature_dim != self.config.action_feature_dim:
            raise ValueError("encoded action_feature_dim mismatch")
        if schema.object_group_count != self.config.object_group_count:
            raise ValueError("encoded object_group_count mismatch")
        if schema.action_ref_feature_dim != self.config.action_ref_feature_dim:
            raise ValueError("encoded action_ref_feature_dim mismatch")
        _check_float_tensor(encoded.state, "state", (self.config.state_dim,))
        _check_float_matrix(encoded.object_features, "object_features", self.config.object_feature_dim, min_rows=1)
        _check_long_vector(encoded.object_card_ids, "object_card_ids", encoded.object_features.shape[0], upper=self.config.card_vocab_size)
        _check_long_vector(encoded.object_groups, "object_groups", encoded.object_features.shape[0], upper=self.config.object_group_count)
        object_count = encoded.object_features.shape[0]
        _check_long_vector(encoded.object_node_ids, "object_node_ids", object_count, upper=object_count)
        if not torch.equal(encoded.object_node_ids, torch.arange(object_count, dtype=torch.long, device=encoded.object_node_ids.device)):
            raise ValueError("object_node_ids must be contiguous local handles")
        _check_float_matrix(encoded.edge_features, "edge_features", self.config.edge_feature_dim, min_rows=0)
        edge_count = encoded.edge_features.shape[0]
        _check_long_vector(encoded.edge_source_indices, "edge_source_indices", edge_count, upper=object_count)
        _check_long_vector(encoded.edge_target_indices, "edge_target_indices", edge_count, upper=object_count)
        _check_float_matrix(encoded.action_features, "action_features", self.config.action_feature_dim, min_rows=1)
        action_count = encoded.action_features.shape[0]
        _check_float_matrix(encoded.action_ref_features, "action_ref_features", self.config.action_ref_feature_dim, min_rows=0)
        ref_count = encoded.action_ref_features.shape[0]
        _check_long_vector(encoded.action_ref_card_ids, "action_ref_card_ids", ref_count, upper=self.config.card_vocab_size)
        _check_long_vector(encoded.action_ref_action_indices, "action_ref_action_indices", ref_count, upper=action_count)
        _check_long_vector(encoded.action_ref_node_indices, "action_ref_node_indices", ref_count, upper=object_count)


def _max_card_token(encoded: EncodedDecision) -> int:
    values = [0]
    if encoded.object_card_ids.numel() > 0:
        values.append(int(torch.max(encoded.object_card_ids).item()))
    if encoded.action_ref_card_ids.numel() > 0:
        values.append(int(torch.max(encoded.action_ref_card_ids).item()))
    return max(values)


def model_config_from_encoded(
    encoded: EncodedDecision,
    *,
    card_vocab_size: int = CARD_TOKEN_VOCAB_SIZE,
    card_embedding_dim: int = MODEL_CARD_EMBEDDING_DIM,
    hidden_dim: int = MODEL_HIDDEN_DIM,
) -> ModelConfig:
    return ModelConfig(
        card_vocab_size=card_vocab_size,
        card_embedding_dim=card_embedding_dim,
        hidden_dim=hidden_dim,
        state_dim=encoded.schema.state_dim,
        object_feature_dim=encoded.schema.object_feature_dim,
        edge_feature_dim=encoded.schema.edge_feature_dim,
        action_feature_dim=encoded.schema.action_feature_dim,
        object_group_count=encoded.schema.object_group_count,
        action_ref_feature_dim=encoded.schema.action_ref_feature_dim,
        feature_schema_version=encoded.schema.version,
        feature_registry_version=encoded.schema.registry_version,
        feature_contract_digest=encoded.schema.contract_digest,
        feature_encoding_digest=encoded.schema.encoding_digest,
    )


def _apply_rowwise(module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] == 0:
        raise ValueError("row-wise module input must be nonempty")
    return torch.cat([module(tensor[i : i + 1]) for i in range(tensor.shape[0])], dim=0)


def _check_float_tensor(tensor: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")


def _check_float_matrix(tensor: torch.Tensor, name: str, width: int, *, min_rows: int) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32")
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} shape mismatch")
    if tensor.shape[0] < min_rows:
        raise ValueError(f"{name} must have at least {min_rows} rows")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")


def _check_long_vector(tensor: torch.Tensor, name: str, length: int, *, upper: int) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if tensor.dtype != torch.long:
        raise ValueError(f"{name} must have dtype torch.long")
    if tensor.ndim != 1 or tensor.shape[0] != length:
        raise ValueError(f"{name} shape mismatch")
    if tensor.numel() == 0:
        return
    if int(torch.min(tensor).item()) < 0:
        raise ValueError(f"{name} contains negative values")
    if int(torch.max(tensor).item()) >= upper:
        raise ValueError(f"{name} contains out-of-range values")


def greedy_action(logits: torch.Tensor) -> int:
    if logits.dtype != torch.float32 or logits.ndim != 1 or logits.shape[0] <= 0:
        raise ValueError("logits must be a nonempty float32 vector")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")
    max_value = torch.max(logits)
    candidates = torch.nonzero(logits == max_value, as_tuple=False).flatten()
    return int(candidates[0].item())
