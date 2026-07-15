"""Small deterministic CPU Torch model for variable legal actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .determinism import configure_torch_determinism
from .features import EncodedDecision


@dataclass(frozen=True)
class ModelConfig:
    card_vocab_size: int = 4096
    card_embedding_dim: int = 16
    hidden_dim: int = 64
    state_dim: int = 0
    object_feature_dim: int = 0
    action_feature_dim: int = 0
    object_group_count: int = 10
    max_action_card_refs: int = 4


class KernelPolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        configure_torch_determinism()
        self.config = config
        self.card_embedding = nn.Embedding(config.card_vocab_size, config.card_embedding_dim, padding_idx=0)
        self.object_encoder = nn.Sequential(
            nn.Linear(config.object_feature_dim + config.card_embedding_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        pooled_dim = config.hidden_dim * config.object_group_count
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim + pooled_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_feature_dim + config.card_embedding_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.Tanh(), nn.Linear(config.hidden_dim, 1))
        self.reset_deterministic_parameters()

    @classmethod
    def from_encoded(cls, encoded: EncodedDecision, *, card_vocab_size: int = 4096, card_embedding_dim: int = 16, hidden_dim: int = 64) -> "KernelPolicyValueNet":
        cfg = ModelConfig(
            card_vocab_size=card_vocab_size,
            card_embedding_dim=card_embedding_dim,
            hidden_dim=hidden_dim,
            state_dim=encoded.schema.state_dim,
            object_feature_dim=encoded.schema.object_feature_dim,
            action_feature_dim=encoded.schema.action_feature_dim,
            object_group_count=encoded.schema.object_group_count,
            max_action_card_refs=encoded.schema.max_action_card_refs,
        )
        return cls(cfg)

    def reset_deterministic_parameters(self) -> None:
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.ndim == 1:
                    values = torch.linspace(-0.05, 0.05, param.numel(), dtype=param.dtype)
                else:
                    values = torch.arange(param.numel(), dtype=param.dtype)
                    values = ((values % 31) - 15) / 200.0
                param.copy_(values.reshape_as(param))
            self.card_embedding.weight[0].zero_()

    def forward(self, encoded: EncodedDecision) -> tuple[torch.Tensor, torch.Tensor]:
        object_card = self.card_embedding(encoded.object_card_ids.clamp(0, self.config.card_vocab_size - 1))
        object_input = torch.cat([encoded.object_features, object_card], dim=-1)
        object_hidden = self.object_encoder(object_input)
        pooled = torch.zeros(
            self.config.object_group_count,
            self.config.hidden_dim,
            dtype=object_hidden.dtype,
            device=object_hidden.device,
        )
        pooled.index_add_(0, encoded.object_groups.clamp(0, self.config.object_group_count - 1), object_hidden)
        pooled_flat = pooled.reshape(-1)
        state_input = torch.cat([encoded.state, pooled_flat], dim=0)
        state_hidden = self.state_encoder(state_input)

        action_emb = self.card_embedding(encoded.action_card_ids.clamp(0, self.config.card_vocab_size - 1))
        mask = (encoded.action_card_ids != 0).to(action_emb.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        action_card = (action_emb * mask).sum(dim=1) / denom
        action_input = torch.cat([encoded.action_features, action_card], dim=-1)
        action_hidden = self.action_encoder(action_input)
        repeated_state = state_hidden.unsqueeze(0).expand(action_hidden.shape[0], -1)
        logits = self.scorer(torch.cat([repeated_state, action_hidden], dim=-1)).squeeze(-1)
        value = self.value_head(state_hidden).squeeze(-1)
        return logits, value


def greedy_action(logits: torch.Tensor) -> int:
    max_value = torch.max(logits)
    candidates = torch.nonzero(logits == max_value, as_tuple=False).flatten()
    return int(candidates[0].item())
