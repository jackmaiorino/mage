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
import os

logger = logging.getLogger(__name__)


class MTGTransformerModel(nn.Module):
    def __init__(self,
                 input_dim: int = 128,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_actions: int = 15,
                 token_vocab: int = 65536,
                 action_vocab: int = 65536,
                 cand_feat_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_actions = num_actions
        self.token_vocab = token_vocab
        self.action_vocab = action_vocab
        self.cand_feat_dim = cand_feat_dim

        # Input scaling and normalization
        # NOTE: Was 0.1, increased to 1.0 to avoid vanishing activations
        self.input_scale = nn.Parameter(torch.ones(1) * 1.0)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Learned (hashed) token IDs for offline card identity features
        self.token_id_emb = nn.Embedding(token_vocab, d_model)

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

        # Candidate-policy head (action-conditional scoring)
        self.action_id_emb = nn.Embedding(action_vocab, d_model)
        self.cand_feat_proj = nn.Sequential(
            nn.Linear(cand_feat_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )
        
        # Cross-attention: candidates attend to full state sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        # Self-attention among candidates for relative comparison
        self.cand_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.cand_self_attn_norm = nn.LayerNorm(d_model)
        
        # MLP-based candidate scorers (concat CLS + attended_candidate → scalar score)
        # Each head learns non-linear interactions between game context and candidate features
        self.policy_scorer = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.policy_scorer_target = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.policy_scorer_card_select = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.policy_scorer_attack = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.policy_scorer_block = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Legacy fixed-action actor head (kept for backwards compatibility / debugging)
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
        self.value_scale = nn.Parameter(torch.tensor(1.0))
        # Start with a slightly lower temperature so small logit differences
        # translate to larger probability gaps and speed up early learning
        self.temperature = nn.Parameter(torch.tensor(0.7))

        # Initialize weights
        self._init_weights()

        # Gradient clipping value (configurable via env var)
        self.max_grad_norm = float(os.getenv('MAX_GRAD_NORM', '1.0'))

        # --- utility lambdas ----------------------------------------
        self._stat_str = lambda a: f"Mean: {a.mean():.4f}, Std: {a.std():.4f}, Min: {a.min():.4f}, Max: {a.max():.4f}"

    def _init_weights(self):
        """Initialize weights with proper gains for gradient flow"""
        # NOTE: Gains were too small (0.005/0.02) causing vanishing gradients
        # Increased to standard ranges for transformer training
        special_actor_linears = {self.actor_proj1, self.actor_proj2}
        special_critic_linears = {self.critic_proj1, self.critic_proj2}
        
        # Collect MLP scorer linears for special initialization
        scorer_linears = set()
        for scorer in [self.policy_scorer, self.policy_scorer_target, self.policy_scorer_card_select,
                       self.policy_scorer_attack, self.policy_scorer_block]:
            for module in scorer.modules():
                if isinstance(module, nn.Linear):
                    scorer_linears.add(module)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m in special_actor_linears:
                    gain = 0.5  # Stronger actor init for clearer policy signal
                elif m in special_critic_linears:
                    gain = 0.5  # Stronger critic init to avoid constant output
                elif m in scorer_linears:
                    gain = 0.5  # MLP scorers get same init as actor (clear scoring signal)
                else:
                    gain = 0.1  # Standard transformer layers (was 0.005)
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, ScaledMultiheadAttention):
                # Proper attention init (was 0.01, too small)
                for param in m.parameters():
                    if param.dim() > 1:  # weight tensors
                        nn.init.xavier_uniform_(param, gain=0.1)
                    else:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.MultiheadAttention):
                # Cross-attention initialization
                for param in m.parameters():
                    if param.dim() > 1:  # weight tensors
                        nn.init.xavier_uniform_(param, gain=0.1)
                    else:
                        nn.init.zeros_(param)

    def forward(self,
                sequences: torch.Tensor,
                masks: torch.Tensor,
                action_masks: Optional[torch.Tensor] = None,
                token_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls = self.encode_state(sequences, masks, token_ids)

        # Process policy and value from CLS embedding
        policy_logits, policy_probs = self._process_policy(cls, action_masks)
        value_scores = self._process_value(cls)

        return policy_logits, policy_probs, value_scores

    def encode_state_full(self,
                          sequences: torch.Tensor,
                          masks: torch.Tensor,
                          token_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (all_tokens [B, S+1, d_model], pad_mask [B, S+1])."""
        x = self.input_proj(sequences)
        if token_ids is not None:
            token_ids = token_ids.clamp(min=0, max=self.token_vocab - 1).long()
            x = x + self.token_id_emb(token_ids)
        x = self.input_norm(x)
        x = x * self.input_scale
        x = self.input_dropout(x)

        cls_tokens = self.cls_token.expand(sequences.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pad_mask = masks.bool()  # True for padding (Java mask uses 1 for pad)
        pad_mask = torch.cat(
            (torch.zeros(sequences.size(0), 1, device=masks.device, dtype=torch.bool), pad_mask), dim=1)

        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=pad_mask)

        return x, pad_mask

    def encode_state(self,
                     sequences: torch.Tensor,
                     masks: torch.Tensor,
                     token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return CLS embedding [B, d_model] for backward compatibility."""
        x, _ = self.encode_state_full(sequences, masks, token_ids)
        return x[:, 0]

    def score_candidates(self,
                         sequences: torch.Tensor,
                         masks: torch.Tensor,
                         token_ids: torch.Tensor,
                         candidate_features: torch.Tensor,
                         candidate_ids: torch.Tensor,
                         candidate_mask: torch.Tensor,
                         head_id: str = "action",
                         pick_index: int = 0,
                         min_targets: int = 0,
                         max_targets: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (candidate_probs, value_scores)."""
        # Encode full state sequence for cross-attention
        state_seq, state_pad_mask = self.encode_state_full(sequences, masks, token_ids)
        cls = state_seq[:, 0]                     # [B, d_model]
        value_scores = self._process_value(cls)

        # Build candidate representations
        cand_feat = self.cand_feat_proj(candidate_features.float())
        cand_ids = candidate_ids.clamp(min=0, max=self.action_vocab - 1).long()
        cand = cand_feat + self.action_id_emb(cand_ids)  # [B, N, d_model]

        # Cross-attention: candidates attend to full state sequence
        attended, _ = self.cross_attn(
            query=cand,
            key=state_seq,
            value=state_seq,
            key_padding_mask=state_pad_mask
        )
        attended = self.cross_attn_norm(attended + cand)   # residual + norm

        # Self-attention among candidates (relative comparison)
        cand_pad_mask = ~candidate_mask.bool()  # True = padding (MHA convention)
        attended_pre = attended
        attended, _ = self.cand_self_attn(
            query=attended,
            key=attended,
            value=attended,
            key_padding_mask=cand_pad_mask
        )
        attended = self.cand_self_attn_norm(attended + attended_pre)  # residual + norm

        # MLP scoring: concat(CLS, attended_candidate) → scalar score
        cls_expanded = cls.unsqueeze(1).expand(-1, attended.size(1), -1)  # [B, N, d_model]
        combined = torch.cat([cls_expanded, attended], dim=-1)             # [B, N, d_model*2]
        
        # Route head selection (ignore pick_index/min/max for now; plumbed for future conditioning)
        hid = str(head_id).strip().lower() if head_id is not None else "action"
        if hid == "target":
            scorer = self.policy_scorer_target
        elif hid == "card_select":
            scorer = self.policy_scorer_card_select
        elif hid == "attack":
            scorer = self.policy_scorer_attack
        elif hid == "block":
            scorer = self.policy_scorer_block
        else:
            scorer = self.policy_scorer
        
        scores = scorer(combined).squeeze(-1)  # [B, N]
        valid = candidate_mask.bool()
        # Numeric safety: if any score is NaN/Inf, treat it as invalid
        scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        scores = scores.masked_fill(~valid, -1e9)

        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Renormalize over valid candidates only; if none valid, return zeros.
        probs = probs * valid.float()
        row_sum = probs.sum(dim=-1, keepdim=True)
        fallback = valid.float() / (valid.float().sum(dim=-1, keepdim=True) + 1e-8)
        probs = torch.where(row_sum > 0, probs / (row_sum + 1e-8), fallback)

        return probs, value_scores

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

    def save(self, path: str, extra_state=None):
        """Save weights & minimal config, plus optional training state."""
        data = {'state_dict': self.state_dict(), 'config': self.get_config()}
        if extra_state:
            data.update(extra_state)
        torch.save(data, path)

    def load(self, path: str):
        """Load weights from file (ignores config—assumes same architecture).
        Returns dict of extra state (optimizer, counters, etc.) if present."""
        ckpt = torch.load(path, map_location='cpu')
        # Backward compatible loads: allow missing keys (e.g., newly added heads).
        res = self.load_state_dict(ckpt['state_dict'], strict=False)
        try:
            if res.missing_keys or res.unexpected_keys:
                logger.warning(
                    "Non-strict load: missing_keys=%d unexpected_keys=%d",
                    len(res.missing_keys), len(res.unexpected_keys)
                )
        except Exception:
            pass
        # Return non-model keys for caller to restore (optimizer, counters, etc.)
        return {k: v for k, v in ckpt.items() if k not in ('state_dict', 'config')}

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

        # Clamp finite logits for numeric stability (do this BEFORE masking so
        # masked positions can stay at a very negative value).
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = torch.clamp(logits, -20.0, 20.0)

        # Mask invalid actions if mask provided
        if action_masks is not None:
            action_masks = action_masks.float()
            logits = logits.masked_fill(~action_masks.bool(), -1e9)

        # Final NaN/Inf guard (must not undo the -1e9 mask)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

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
        """Return scalar state-value from critic head."""

        # Normalisation & first projection
        x = self.critic_norm(x)
        x = self.critic_proj1(x)
        x = F.relu(x)
        x = self.critic_norm1(x)

        # Second projection → scalar
        value = self.critic_proj2(x)

        # Numeric safety
        value = torch.nan_to_num(value, nan=0.0, posinf=50.0, neginf=-50.0)
        value = torch.clamp(value, -50.0, 50.0)

        # Apply learnable scale (no tanh: allow critic to represent targets > 1.0)
        scale = torch.clamp(self.value_scale, 0.01, 10.0)
        value = value * scale
        value = torch.clamp(value, -10.0, 10.0)

        # Final safety clamp for numerical stability
        if torch.isnan(value).any() or torch.isinf(value).any():
            logger.error("NaN/Inf in value head – replacing with zeros")
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return value

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
        # NOTE: Was 0.1, increased to 1.0 to avoid uniform attention
        self.scale = nn.Parameter(torch.ones(1) * 1.0)

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
