import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional
import logging
import time
import traceback
import math

logger = logging.getLogger(__name__)


class MTGTransformerModel(nn.Module):
    def __init__(self,
                 input_dim: int = 128,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_actions: int = 15):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_actions = num_actions

        # Input scaling and normalization
        self.input_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Create transformer encoder layers with our custom attention
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])

        # Replace the attention mechanism in each layer
        for layer in self.transformer_layers:
            layer.self_attn = ScaledMultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )

        # CLS token with small initialization
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.01)

        # Actor head with proper scaling
        self.actor_norm = nn.LayerNorm(d_model)
        self.actor_proj1 = nn.Linear(d_model, d_model // 2)
        self.actor_norm1 = nn.LayerNorm(d_model // 2)
        self.actor_proj2 = nn.Linear(d_model // 2, num_actions)

        # Critic head with proper scaling
        self.critic_norm = nn.LayerNorm(d_model)
        self.critic_proj1 = nn.Linear(d_model, d_model // 2)
        self.critic_norm1 = nn.LayerNorm(d_model // 2)
        self.critic_proj2 = nn.Linear(d_model // 2, 1)

        # Learnable scaling factors
        self.value_scale = nn.Parameter(torch.tensor(0.1))
        # Start with a higher temperature so initial logits are small → soft probabilities
        self.temperature = nn.Parameter(torch.tensor(5.0))

        # Initialize weights
        self._init_weights()

        # Gradient clipping value
        self.max_grad_norm = 1.0

        # --- utility lambdas ----------------------------------------
        self._stat_str = lambda a: f"Mean: {a.mean():.4f}, Std: {a.std():.4f}, Min: {a.min():.4f}, Max: {a.max():.4f}"

    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use smaller gain to avoid large initial activations
                nn.init.xavier_uniform_(m.weight, gain=0.005)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, ScaledMultiheadAttention):
                # Re-initialize attention weights with small gain
                for param in m.parameters():
                    if param.dim() > 1:  # weight tensors
                        nn.init.xavier_uniform_(param, gain=0.01)
                    else:
                        nn.init.zeros_(param)

    def forward(self,
                sequences: torch.Tensor,
                masks: torch.Tensor,
                action_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input projection and normalization
        x = self.input_proj(sequences)
        x = self.input_norm(x)
        x = x * self.input_scale
        x = self.input_dropout(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(sequences.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Extend mask for CLS token
        extended_mask = torch.cat(
            (torch.ones(sequences.size(0), 1, device=masks.device, dtype=torch.bool), masks), dim=1)

        # Process through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=~extended_mask.bool())

        # Get CLS token output
        x = x[:, 0]

        # Process policy and value
        policy_logits, policy_probs = self._process_policy(x, action_masks)
        value_scores = self._process_value(x)

        return policy_logits, policy_probs, value_scores

    def predict_batch(self,
                      sequences: np.ndarray,
                      masks: np.ndarray,
                      action_masks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for Java interface with input validation

        Args:
            sequences: Batch of state sequences [batch_size, seq_len, d_model]
            masks: Attention masks [batch_size, seq_len]
            action_masks: Optional masks for valid actions [batch_size, num_actions]

        Returns:
            Tuple of (action_probs, value_scores) as numpy arrays
        """
        self.eval()
        with torch.no_grad():
            # Sanitize inputs
            sequences = self._sanitize_numpy("input sequences", sequences)
            masks = self._sanitize_numpy("input masks", masks)
            if action_masks is not None:
                action_masks = self._sanitize_numpy(
                    "action masks", action_masks)

            # Convert inputs to tensors and move to same device as model
            device = next(self.parameters()).device
            sequences = torch.FloatTensor(sequences).to(device)
            masks = torch.BoolTensor(masks).to(device)
            if action_masks is not None:
                action_masks = torch.FloatTensor(action_masks).to(device)

            # Validate tensor shapes
            logger.info(f"Input shapes - Sequences: {sequences.shape}, "
                        f"Masks: {masks.shape}, "
                        f"Action masks: {action_masks.shape if action_masks is not None else None}")

            # Get predictions – ignore logits here
            _logits, action_probs, value_scores = self.forward(
                sequences, masks, action_masks)

            # Sanitize outputs
            value_scores = self._sanitize_tensor("value scores", value_scores)
            value_scores = torch.clamp(value_scores, -1.0, 1.0)

            # Convert to numpy and validate again
            action_probs_np = action_probs.cpu().numpy()
            value_scores_np = value_scores.cpu().numpy()

            # Final validation of numpy arrays
            value_scores_np = self._sanitize_numpy(
                "value scores np", value_scores_np)
            value_scores_np = np.clip(value_scores_np, -1.0, 1.0)

            # Log final prediction statistics
            logger.info(f"Final prediction stats - Action probs mean: {np.mean(action_probs_np):.4f}, "
                        f"Value scores mean: {np.mean(value_scores_np):.4f}, "
                        f"Value scores min: {np.min(value_scores_np):.4f}, "
                        f"Value scores max: {np.max(value_scores_np):.4f}")

            return action_probs_np, value_scores_np

    def save(self, path: str):
        """Save weights & minimal config."""
        torch.save({'state_dict': self.state_dict(),
                   'config': self.get_config()}, path)

    def load(self, path: str):
        """Load weights from file (ignores config—assumes same architecture)."""
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt['state_dict'])

    # ------------------------------------------------------------------
    # Helper functions for policy and value heads (actor / critic)
    # ------------------------------------------------------------------

    def _compute_policy_logits(self, x: torch.Tensor, action_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return raw (un-softmaxed) logits for the actor head."""

        # Normalisation & first projection
        x = self.actor_norm(x)
        x = self.actor_proj1(x)
        x = F.relu(x)
        x = self.actor_norm1(x)

        # Second projection → logits
        logits = self.actor_proj2(x)
        # Replace NaN/Inf in logits to avoid invalid softmax
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        # Temperature scaling (learnable)
        temp = torch.clamp(self.temperature, min=0.1)
        logits = logits / (temp + 1e-8)

        # Mask invalid actions if mask provided
        if action_masks is not None:
            action_masks = action_masks.float()
            logits = logits.masked_fill(~action_masks.bool(), -1e9)

        # Final NaN/Inf guard and numeric clamp before softmax
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = torch.clamp(logits, -20.0, 20.0)

        return logits

    def _process_policy(self, x: torch.Tensor, action_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, probabilities) for the current actor head."""

        logits = self._compute_policy_logits(x, action_masks)
        probs = F.softmax(logits, dim=-1)

        # Numerical safety
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            logger.error(
                "NaN/Inf in policy probs – replacing with uniform distribution")
            probs = torch.ones_like(probs) / self.num_actions

        # Ensure sum to one
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        return logits, probs

    def _process_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return bounded scalar state-value from critic head."""

        # Normalisation & first projection
        x = self.critic_norm(x)
        x = self.critic_proj1(x)
        x = F.relu(x)
        x = self.critic_norm1(x)

        # Second projection → scalar
        value = self.critic_proj2(x)

        # Pre-activation clamp to avoid extreme values that could produce NaNs/Inf
        value = torch.nan_to_num(value, nan=0.0, posinf=50.0, neginf=-50.0)
        value = torch.clamp(value, -50.0, 50.0)

        # Bound output to [-1, 1]
        value = torch.tanh(value)

        # Sanitize and apply learnable scaling factor
        safe_scale = torch.nan_to_num(
            self.value_scale, nan=0.0, posinf=5.0, neginf=-5.0)
        # Keep scale in a reasonable range to avoid extreme shrink/expand
        safe_scale = torch.clamp(safe_scale, -5.0, 5.0)
        value = value * torch.sigmoid(safe_scale)

        # Final safety clamp
        if torch.isnan(value).any() or torch.isinf(value).any():
            logger.error(
                "NaN/Inf in value head – clamping to finite range after tanh")
            value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)

        return torch.clamp(value, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nan_to_num(arr: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with safe numbers (in-place safe)."""
        if np.isnan(arr).any() or np.isinf(arr).any():
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _sanitize_numpy(self, name: str, arr: np.ndarray) -> np.ndarray:
        """Detect & repair NaN/Inf, logging a single message."""
        if np.isnan(arr).any() or np.isinf(arr).any():
            logger.error(
                "NaN/Inf detected in %s – replacing with zeros. [%s]", name, self._stat_str(arr))
            arr = self._nan_to_num(arr)
        return arr

    def _sanitize_tensor(self, name: str, t: torch.Tensor) -> torch.Tensor:
        if torch.isnan(t).any() or torch.isinf(t).any():
            logger.error(
                "NaN/Inf detected in %s – clamping. [%s]", name, self._stat_str(t.cpu()))
            t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
        return t

    # Small util for configuration persistence
    def get_config(self):
        return dict(input_dim=self.input_dim, d_model=self.d_model, nhead=self.transformer_layers[0].self_attn.num_heads,
                    num_layers=len(self.transformer_layers), dim_feedforward=self.transformer_layers[0].linear1.out_features,
                    dropout=self.transformer_layers[0].dropout.p, num_actions=self.num_actions)


class ScaledMultiheadAttention(nn.MultiheadAttention):
    """Multi-head attention with a learnable pre-scaling of queries & keys.

    Subclassing ``nn.MultiheadAttention`` means we automatically present the
    full public API (e.g. ``in_proj_weight``, ``_qkv_same_embed_dim``) expected
    by upstream PyTorch components such as ``nn.TransformerEncoderLayer``. The
    only behavioural change is a learnable scalar factor applied to *Q* and *K*
    before the dot-product attention is computed.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, batch_first: bool = False):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads,
                         dropout=dropout, batch_first=batch_first)

        # Learnable scale shared across heads.
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    # ------------------------------------------------------------------
    # Override forward to inject scaling, then delegate to super().forward
    # ------------------------------------------------------------------

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, is_causal=False):
        # Apply learnable scaling to queries and keys
        q = query * self.scale
        k = key * self.scale

        # Handle the optional is_causal argument depending on torch version
        forward_kwargs = {
            'key_padding_mask': key_padding_mask,
            'need_weights': need_weights,
            'attn_mask': attn_mask,
        }
        if 'is_causal' in super().forward.__code__.co_varnames:
            forward_kwargs['is_causal'] = is_causal

        return super().forward(q, k, value, **forward_kwargs)

    def _get_name(self):
        return 'ScaledMultiheadAttention'
