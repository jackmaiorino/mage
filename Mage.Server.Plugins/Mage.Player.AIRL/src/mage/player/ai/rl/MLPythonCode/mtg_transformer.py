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
        self.value_use_separate_critic = os.getenv(
            "VALUE_USE_SEPARATE_CRITIC_ENCODER", "0").strip().lower() in ("1", "true", "yes", "on")
        self.candidate_q_blend = float(os.getenv("CANDIDATE_Q_BLEND", "0.0"))
        self.candidate_q_blend_min_top_q = float(os.getenv("CANDIDATE_Q_BLEND_MIN_TOP_Q", "-1.0"))
        self.candidate_q_blend_min_margin = float(os.getenv("CANDIDATE_Q_BLEND_MIN_MARGIN", "0.0"))
        self.candidate_q_blend_heads = self._parse_candidate_q_blend_heads(
            os.getenv("CANDIDATE_Q_BLEND_HEADS", "*"))

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
        self.policy_scorer_mulligan = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.candidate_q_scorer = nn.Sequential(
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

        # --- Separate critic encoder (independent from policy encoder) ---
        self.critic_input_proj = nn.Linear(input_dim, d_model)
        self.critic_input_norm = nn.LayerNorm(d_model)
        # Shares self.token_id_emb (card identity is objective, not policy-dependent)
        self.critic_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.critic_cls_token, std=0.01)
        critic_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        critic_layer.self_attn = ScaledMultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.critic_transformer = nn.ModuleList([critic_layer])

        # Critic MLP head (on top of critic encoder's CLS)
        self.critic_norm = nn.LayerNorm(d_model)
        self.critic_proj1 = nn.Linear(d_model, d_model // 2)
        self.critic_norm1 = nn.LayerNorm(d_model // 2)
        self.critic_proj2 = nn.Linear(d_model // 2, 1)

        # Belief head (Phase 1 aux loss): classifier over deck archetypes.
        # Input: shared encoder CLS. Output: logits over NUM_ARCHETYPES classes
        # (0=Wildfire, 1=Rally, 2=Affinity, 3=Elves, 4=SpyCombo, 5=Burn,
        # 6=Terror, 7=CawGates, 8=Faeries). Trained only on the
        # public state the policy sees -- it must INFER the archetype, not
        # be told it. At inference, softmax(belief_head(cls)) gives the
        # archetype distribution for ISMCTS determinization sampling.
        self.num_archetypes = int(os.getenv('NUM_ARCHETYPES', '9'))
        self.belief_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, self.num_archetypes),
        )
        self.card_belief_dim = max(0, int(os.getenv('CARD_BELIEF_DIM', '0')))
        self.card_belief_head = None
        if self.card_belief_dim > 0:
            self.card_belief_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, self.card_belief_dim),
            )

        # Learnable scaling factors
        self.value_scale = nn.Parameter(torch.tensor(1.0))
        # Start with a slightly lower temperature so small logit differences
        # translate to larger probability gaps and speed up early learning
        self.temperature = nn.Parameter(torch.tensor(0.7))

        # Initialize weights
        self._init_weights()

        # Gradient clipping value (configurable via env var)
        self.max_grad_norm = float(os.getenv('MAX_GRAD_NORM', '1.0'))

        # Temperature floor: prevents policy from becoming near-deterministic
        self.temperature_floor = float(os.getenv('TEMPERATURE_FLOOR', '0.3'))

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
                       self.policy_scorer_attack, self.policy_scorer_block, self.policy_scorer_mulligan,
                       self.candidate_q_scorer]:
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

        # Process policy from shared encoder CLS. The value path is selected
        # by VALUE_USE_SEPARATE_CRITIC_ENCODER for critic-isolation runs.
        policy_logits, policy_probs = self._process_policy(cls, action_masks)
        value_scores = self._process_value(sequences, masks, token_ids)

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

    def encode_state_critic(self,
                            sequences: torch.Tensor,
                            masks: torch.Tensor,
                            token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run separate critic encoder, return critic CLS [B, d_model]."""
        x = self.critic_input_proj(sequences)
        if token_ids is not None:
            token_ids = token_ids.clamp(min=0, max=self.token_vocab - 1).long()
            x = x + self.token_id_emb(token_ids)  # shared embedding
        x = self.critic_input_norm(x)
        x = x * self.input_scale  # reuse learned input scale

        cls_tokens = self.critic_cls_token.expand(sequences.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pad_mask = masks.bool()
        pad_mask = torch.cat(
            (torch.zeros(sequences.size(0), 1, device=masks.device, dtype=torch.bool), pad_mask), dim=1)

        for layer in self.critic_transformer:
            x = layer(x, src_key_padding_mask=pad_mask)

        return x[:, 0]  # critic CLS

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
                         max_targets: int = 0,
                         return_cls: bool = False,
                         return_logits: bool = False,
                         return_candidate_q: bool = False):
        """Return (candidate_probs, value_scores). If return_cls, also returns the shared-encoder CLS as the 3rd element."""
        # Encode full state sequence for cross-attention and candidate policy.
        state_seq, state_pad_mask = self.encode_state_full(sequences, masks, token_ids)
        cls = state_seq[:, 0]                     # [B, d_model]
        if self.value_use_separate_critic:
            value_cls = self.encode_state_critic(sequences, masks, token_ids)
        else:
            value_cls = cls
        value_scores = self._process_value_from_cls(value_cls)

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
        elif hid == "mulligan":
            scorer = self.policy_scorer_mulligan
        else:
            scorer = self.policy_scorer
        
        scores = scorer(combined).squeeze(-1)  # [B, N]
        candidate_q = torch.tanh(self.candidate_q_scorer(combined).squeeze(-1))  # [B, N], terminal-return scale
        valid = candidate_mask.bool()
        # Numeric safety: if any score is NaN/Inf, treat it as invalid
        scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        candidate_q = torch.nan_to_num(candidate_q, nan=0.0, posinf=1.0, neginf=-1.0)
        candidate_q = candidate_q.masked_fill(~valid, 0.0)
        if self.candidate_q_blend != 0.0 and self._candidate_q_blend_enabled_for_head(hid):
            scores = scores + self._candidate_q_blend_bonus(candidate_q, valid)
        scores = scores.masked_fill(~valid, -1e9)

        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Renormalize over valid candidates only; if none valid, return zeros.
        probs = probs * valid.float()
        row_sum = probs.sum(dim=-1, keepdim=True)
        fallback = valid.float() / (valid.float().sum(dim=-1, keepdim=True) + 1e-8)
        probs = torch.where(row_sum > 0, probs / (row_sum + 1e-8), fallback)

        result = [probs, value_scores]
        if return_cls:
            result.append(cls)
        if return_logits:
            result.append(scores)
        if return_candidate_q:
            result.append(candidate_q)
        return tuple(result)

    def _candidate_q_blend_bonus(self, candidate_q: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        blend = float(getattr(self, "candidate_q_blend", 0.0))
        if blend == 0.0:
            return torch.zeros_like(candidate_q)
        min_top_q = float(getattr(self, "candidate_q_blend_min_top_q", -1.0))
        min_margin = float(getattr(self, "candidate_q_blend_min_margin", 0.0))
        if min_top_q <= -1.0 and min_margin <= 0.0:
            return blend * candidate_q

        valid_q = candidate_q.masked_fill(~valid, -1e9)
        top_q = valid_q.max(dim=-1).values
        gate = torch.ones_like(top_q, dtype=torch.bool)
        if min_top_q > -1.0:
            gate = gate & (top_q >= min_top_q)
        if min_margin > 0.0:
            if candidate_q.size(-1) > 1:
                top2 = torch.topk(valid_q, k=2, dim=-1).values
                margin = top2[:, 0] - top2[:, 1]
            else:
                margin = torch.zeros_like(top_q)
            gate = gate & (margin >= min_margin)
        return blend * candidate_q * gate.to(candidate_q.dtype).unsqueeze(-1)

    @staticmethod
    def _parse_candidate_q_blend_heads(raw: str):
        text = str(raw or "*").strip().lower()
        if text in ("", "*", "all"):
            return None
        if text in ("none", "off", "false", "0"):
            return set()
        aliases = {
            "actions": "action",
            "default": "action",
            "spell": "action",
            "spells": "action",
            "ability": "action",
            "abilities": "action",
            "targets": "target",
            "select_targets": "target",
            "select_target": "target",
            "card": "card_select",
            "cards": "card_select",
            "select_card": "card_select",
            "card_select": "card_select",
            "attacks": "attack",
            "declare_attacks": "attack",
            "blocks": "block",
            "declare_blocks": "block",
            "mulligans": "mulligan",
            "london_mulligan": "mulligan",
        }
        heads = set()
        for part in text.replace(";", ",").split(","):
            token = part.strip().lower()
            if token:
                heads.add(aliases.get(token, token))
        return heads

    def _candidate_q_blend_enabled_for_head(self, head_id: str) -> bool:
        heads = getattr(self, "candidate_q_blend_heads", None)
        if heads is None:
            return True
        hid = str(head_id or "action").strip().lower()
        return hid in heads

    def belief_logits_from_cls(self, cls: torch.Tensor) -> torch.Tensor:
        """Archetype classifier logits from shared-encoder CLS.

        Args:
            cls: [B, d_model] shared-encoder CLS embedding.

        Returns:
            [B, num_archetypes] logits over deck archetypes.
        """
        return self.belief_head(cls)

    def card_belief_logits_from_cls(self, cls: torch.Tensor) -> torch.Tensor:
        """Hidden-card count logits from shared-encoder CLS."""
        if self.card_belief_head is None:
            return cls.new_zeros((cls.shape[0], 0))
        return self.card_belief_head(cls)

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
        state = ckpt['state_dict']
        own_state = self.state_dict()
        filtered_state = {}
        skipped_incompatible = []
        for key, value in state.items():
            if key in own_state:
                try:
                    if tuple(value.shape) != tuple(own_state[key].shape):
                        skipped_incompatible.append(key)
                        continue
                except Exception:
                    pass
            filtered_state[key] = value
        # Backward compatible loads: allow missing keys (e.g., newly added or widened heads).
        res = self.load_state_dict(filtered_state, strict=False)
        try:
            if res.missing_keys or res.unexpected_keys or skipped_incompatible:
                logger.warning(
                    "Non-strict load: missing_keys=%d unexpected_keys=%d skipped_incompatible=%d",
                    len(res.missing_keys), len(res.unexpected_keys), len(skipped_incompatible)
                )
        except Exception:
            pass
        # Return non-model keys for caller to restore (optimizer, counters, etc.)
        extra = {k: v for k, v in ckpt.items() if k not in ('state_dict', 'config')}
        if skipped_incompatible:
            extra['_skipped_incompatible_state_keys'] = skipped_incompatible
        return extra

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

        # Temperature scaling (learnable); floor prevents near-deterministic collapse
        temp = torch.clamp(self.temperature, min=self.temperature_floor)
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

    def _process_value_from_cls(self, cls: torch.Tensor) -> torch.Tensor:
        """Scalar value from a shared-policy or separate-critic CLS feature."""
        x = self.critic_norm(cls)
        x = self.critic_proj1(x)
        x = F.relu(x)
        x = self.critic_norm1(x)
        value = self.critic_proj2(x)
        value = torch.nan_to_num(value, nan=0.0, posinf=50.0, neginf=-50.0)
        value = torch.clamp(value, -50.0, 50.0)
        scale = torch.clamp(self.value_scale, 0.01, 10.0)
        value = value * scale
        value = torch.clamp(value, -10.0, 10.0)
        if torch.isnan(value).any() or torch.isinf(value).any():
            logger.error("NaN/Inf in value head – replacing with zeros")
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return value

    def _process_value(self,
                       sequences: torch.Tensor,
                       masks: torch.Tensor,
                       token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Value wrapper used by forward and legacy callers."""
        if self.value_use_separate_critic:
            cls = self.encode_state_critic(sequences, masks, token_ids)
        else:
            cls = self.encode_state(sequences, masks, token_ids)
        return self._process_value_from_cls(cls)

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
                    dropout=self.transformer_layers[0].dropout.p, num_actions=self.num_actions,
                    card_belief_dim=self.card_belief_dim)


class SingleHeadScorer(nn.Module):
    """Wraps MTGTransformerModel.score_candidates for a single head for ONNX export.

    Inlines the score_candidates logic with frozen head selection and no
    data-dependent control flow (if/elif on head_id, if isnan checks), so
    torch.onnx.export can trace the graph cleanly.
    """
    def __init__(self, model: 'MTGTransformerModel', head_id: str):
        super().__init__()
        self.model = model
        # Resolve the scorer at construction time (not in forward)
        hid = head_id.lower().strip()
        if hid == "target":
            self.scorer = model.policy_scorer_target
        elif hid == "card_select":
            self.scorer = model.policy_scorer_card_select
        elif hid == "attack":
            self.scorer = model.policy_scorer_attack
        elif hid == "block":
            self.scorer = model.policy_scorer_block
        elif hid == "mulligan":
            self.scorer = model.policy_scorer_mulligan
        else:
            self.scorer = model.policy_scorer
        self.head_id = hid
        self.candidate_q_blend = float(getattr(model, "candidate_q_blend", 0.0))

    @staticmethod
    def _unfused_encoder_layer(layer, x, src_key_padding_mask):
        """Run TransformerEncoderLayer without the fused C++ op (for ONNX export)."""
        # Self-attention
        x2, _ = layer.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = layer.norm1(x + layer.dropout1(x2))
        # Feed-forward
        x2 = layer.linear2(layer.dropout(F.relu(layer.linear1(x))))
        x = layer.norm2(x + layer.dropout2(x2))
        return x

    def _encode_state_unfused(self, sequences, masks, token_ids):
        """encode_state_full but with unfused transformer layers for ONNX export."""
        m = self.model
        x = m.input_proj(sequences) * m.input_scale
        if token_ids is not None:
            safe_ids = token_ids.clamp(min=0, max=m.token_id_emb.num_embeddings - 1)
            x = x + m.token_id_emb(safe_ids)
        x = m.input_norm(x)
        cls_expanded = m.cls_token.expand(sequences.size(0), -1, -1)
        x = torch.cat((cls_expanded, x), dim=1)
        pad_mask = masks.bool()
        pad_mask = torch.cat(
            (torch.zeros(sequences.size(0), 1, device=masks.device, dtype=torch.bool), pad_mask), dim=1)
        for layer in m.transformer_layers:
            x = self._unfused_encoder_layer(layer, x, pad_mask)
        return x, pad_mask

    def _encode_state_critic_unfused(self, sequences, masks, token_ids):
        """Critic encoder with unfused transformer layers for ONNX export."""
        m = self.model
        x = m.critic_input_proj(sequences)
        if token_ids is not None:
            safe_ids = token_ids.clamp(min=0, max=m.token_id_emb.num_embeddings - 1)
            x = x + m.token_id_emb(safe_ids)
        x = m.critic_input_norm(x)
        x = x * m.input_scale
        cls_expanded = m.critic_cls_token.expand(sequences.size(0), -1, -1)
        x = torch.cat((cls_expanded, x), dim=1)
        pad_mask = masks.bool()
        pad_mask = torch.cat(
            (torch.zeros(sequences.size(0), 1, device=masks.device, dtype=torch.bool), pad_mask), dim=1)
        for layer in m.critic_transformer:
            x = self._unfused_encoder_layer(layer, x, pad_mask)
        return x[:, 0]  # critic CLS

    def forward(self, sequences, masks, token_ids, cand_features, cand_ids, cand_mask):
        m = self.model
        # Encode policy state (unfused for ONNX export).
        state_seq, state_pad_mask = self._encode_state_unfused(sequences, masks, token_ids)
        cls = state_seq[:, 0]

        if m.value_use_separate_critic:
            value_cls = self._encode_state_critic_unfused(sequences, masks, token_ids)
        else:
            value_cls = cls

        # Value head: MLP on top of selected value CLS.
        v = m.critic_norm(value_cls)
        v = m.critic_proj1(v)
        v = torch.relu(v)
        v = m.critic_norm1(v)
        v = m.critic_proj2(v)
        v = torch.nan_to_num(v, nan=0.0, posinf=50.0, neginf=-50.0)
        v = torch.clamp(v, -50.0, 50.0)
        scale = torch.clamp(m.value_scale, 0.01, 10.0)
        v = v * scale
        value_scores = torch.clamp(v, -10.0, 10.0)

        # Candidate embedding
        cand_feat = m.cand_feat_proj(cand_features.float())
        cand_ids_clamped = cand_ids.clamp(min=0, max=m.action_vocab - 1).long()
        cand = cand_feat + m.action_id_emb(cand_ids_clamped)

        # Cross-attention: candidates <-> state
        attended, _ = m.cross_attn(query=cand, key=state_seq, value=state_seq,
                                   key_padding_mask=state_pad_mask)
        attended = m.cross_attn_norm(attended + cand)

        # Self-attention among candidates
        cand_pad_mask = ~cand_mask.bool()
        attended_pre = attended
        attended, _ = m.cand_self_attn(query=attended, key=attended, value=attended,
                                       key_padding_mask=cand_pad_mask)
        attended = m.cand_self_attn_norm(attended + attended_pre)

        # Score with frozen head
        cls_expanded = cls.unsqueeze(1).expand(-1, attended.size(1), -1)
        combined = torch.cat([cls_expanded, attended], dim=-1)
        scores = self.scorer(combined).squeeze(-1)
        candidate_q = torch.tanh(m.candidate_q_scorer(combined).squeeze(-1))

        # Mask + softmax (no data-dependent if-checks)
        valid = cand_mask.bool()
        scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        candidate_q = torch.nan_to_num(candidate_q, nan=0.0, posinf=1.0, neginf=-1.0)
        candidate_q = candidate_q.masked_fill(~valid, 0.0)
        if self.candidate_q_blend != 0.0 and m._candidate_q_blend_enabled_for_head(self.head_id):
            scores = scores + m._candidate_q_blend_bonus(candidate_q, valid)
        scores = scores.masked_fill(~valid, -1e9)
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs * valid.float()
        row_sum = probs.sum(dim=-1, keepdim=True)
        fallback = valid.float() / (valid.float().sum(dim=-1, keepdim=True) + 1e-8)
        probs = torch.where(row_sum > 0, probs / (row_sum + 1e-8), fallback)

        return probs, value_scores


class ScaledMultiheadAttention(nn.MultiheadAttention):
    """Multi-head attention with a learnable pre-scaling of queries & keys.

    Subclassing ``nn.MultiheadAttention`` keeps the public parameter surface
    (``in_proj_weight``, ``in_proj_bias``, ``out_proj``, etc.) so existing
    checkpoints load unchanged. The forward path, however, invokes
    ``F.scaled_dot_product_attention`` directly so that PyTorch picks the
    fused flash / memory-efficient kernel -- the attention matrix is never
    materialized, which is the dominant memory cost on long sequences.

    Set env var ``USE_FUSED_SDPA=0`` to fall back to the parent's forward
    for A/B comparison.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, batch_first: bool = False):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads,
                         dropout=dropout, batch_first=batch_first)

        # Learnable scale shared across heads.
        # NOTE: Was 0.1, increased to 1.0 to avoid uniform attention
        self.scale = nn.Parameter(torch.ones(1) * 1.0)

    def _effective_scale(self):
        # Learnable scale has collapsed to 0 in every saved checkpoint, which
        # makes q*scale == k*scale == 0 and forces uniform attention (bag-of-
        # words). Floor at a positive value so attention remains meaningful and
        # the rest of the network has a chance to recover; gradient still lets
        # the scale grow above the floor. Override via env for ablation.
        floor = float(os.getenv("SCALED_MHA_MIN_SCALE", "1.0"))
        return torch.clamp(self.scale, min=floor)

    def _parent_forward(self, query, key, value, key_padding_mask, need_weights, attn_mask, is_causal):
        s = self._effective_scale()
        q = query * s
        k = key * s
        forward_kwargs = {
            'key_padding_mask': key_padding_mask,
            'need_weights': need_weights,
            'attn_mask': attn_mask,
        }
        if 'is_causal' in super().forward.__code__.co_varnames:
            forward_kwargs['is_causal'] = is_causal
        return super().forward(q, k, value, **forward_kwargs)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, is_causal=False):
        # Fall back to parent when: caller needs attention weights, non-default
        # embed-dim layout, non-3D tensors, or explicitly disabled via env.
        _fused_enabled = os.getenv("USE_FUSED_SDPA", "1") != "0"
        if (not _fused_enabled) or need_weights or (not self._qkv_same_embed_dim) \
                or query.dim() != 3 or (not self.batch_first) or attn_mask is not None:
            return self._parent_forward(query, key, value, key_padding_mask, need_weights, attn_mask, is_causal)

        E = self.embed_dim
        H = self.num_heads
        Hd = self.head_dim
        B, N_q, _ = query.shape
        N_k = key.shape[1]

        # Apply learnable scale (floored to avoid the collapsed-to-0 failure
        # mode) to query & key *inputs* pre-projection, matching the original
        # super().forward(q*scale, k*scale, value) semantics.
        s = self._effective_scale()
        q_in = query * s
        k_in = key * s

        if self.in_proj_bias is not None:
            q_proj = F.linear(q_in, self.in_proj_weight[:E], self.in_proj_bias[:E])
            k_proj = F.linear(k_in, self.in_proj_weight[E:2 * E], self.in_proj_bias[E:2 * E])
            v_proj = F.linear(value, self.in_proj_weight[2 * E:], self.in_proj_bias[2 * E:])
        else:
            q_proj = F.linear(q_in, self.in_proj_weight[:E])
            k_proj = F.linear(k_in, self.in_proj_weight[E:2 * E])
            v_proj = F.linear(value, self.in_proj_weight[2 * E:])

        # [B, N, E] -> [B, H, N, Hd]
        q = q_proj.view(B, N_q, H, Hd).transpose(1, 2)
        k = k_proj.view(B, N_k, H, Hd).transpose(1, 2)
        v = v_proj.view(B, N_k, H, Hd).transpose(1, 2)

        # key_padding_mask: bool [B, N_k]; True = padding. SDPA attn_mask float:
        # 0 = attend, -inf = mask out. Broadcast to [B, 1, 1, N_k].
        sdpa_mask = None
        if key_padding_mask is not None:
            kpm = key_padding_mask.bool()
            sdpa_mask = torch.zeros(B, 1, 1, N_k, dtype=q.dtype, device=q.device)
            sdpa_mask = sdpa_mask.masked_fill(kpm.unsqueeze(1).unsqueeze(2), float('-inf'))

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, N_q, E)
        out = self.out_proj(out)
        return out, None

    def _get_name(self):
        return 'ScaledMultiheadAttention'
