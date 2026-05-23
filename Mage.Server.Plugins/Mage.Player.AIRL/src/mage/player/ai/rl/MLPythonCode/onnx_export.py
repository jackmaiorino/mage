#!/usr/bin/env python3
"""Export MTGTransformerModel to ONNX -- one file per policy head.

Usage:
    py -3.12 MLPythonCode/onnx_export.py --model-path profiles/Pauper-Rally-A/models/model_latest.pt --output-dir /tmp/onnx_test/
"""
import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from mtg_transformer import MTGTransformerModel, SingleHeadScorer

HEAD_IDS = ["action", "target", "card_select", "attack", "block", "mulligan"]


class BeliefHeadScorer(nn.Module):
    """ONNX wrapper: run shared encoder, emit archetype logits.

    Input: (sequences, masks, token_ids) — same signature as action heads,
    so the Java side can build tensors once and call either model.
    Output: logits [B, num_archetypes] in the Java archetype-label order.
    """

    def __init__(self, model: 'MTGTransformerModel'):
        super().__init__()
        self.model = model

    @staticmethod
    def _unfused_encoder_layer(layer, x, src_key_padding_mask):
        """Mirror SingleHeadScorer._unfused_encoder_layer so ONNX export avoids
        PyTorch's fused multi-head attention op (which the exporter can't handle)."""
        x2, _ = layer.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = layer.norm1(x + layer.dropout1(x2))
        x2 = layer.linear2(layer.dropout(F.relu(layer.linear1(x))))
        x = layer.norm2(x + layer.dropout2(x2))
        return x

    def forward(self, sequences, masks, token_ids):
        m = self.model
        x = m.input_proj(sequences) * m.input_scale
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
        cls = x[:, 0]
        return m.belief_head(cls)  # [B, num_archetypes]


class ManualMHA(nn.Module):
    """ONNX-exportable replacement for nn.MultiheadAttention.

    Uses explicit linear projections + scaled dot-product attention
    instead of the fused _native_multi_head_attention kernel that
    PyTorch's ONNX exporter can't handle.
    """

    def __init__(self, src: nn.MultiheadAttention):
        super().__init__()
        self.embed_dim = src.embed_dim
        self.num_heads = src.num_heads
        self.head_dim = src.embed_dim // src.num_heads
        self.batch_first = src.batch_first

        # Copy projection weights. nn.MultiheadAttention stores Q/K/V
        # in a single in_proj_weight [3*E, E] + in_proj_bias [3*E].
        E = self.embed_dim
        self.q_proj = nn.Linear(E, E, bias=src.in_proj_bias is not None)
        self.k_proj = nn.Linear(E, E, bias=src.in_proj_bias is not None)
        self.v_proj = nn.Linear(E, E, bias=src.in_proj_bias is not None)

        with torch.no_grad():
            if src._qkv_same_embed_dim:
                w = src.in_proj_weight
                self.q_proj.weight.copy_(w[:E])
                self.k_proj.weight.copy_(w[E:2*E])
                self.v_proj.weight.copy_(w[2*E:])
                if src.in_proj_bias is not None:
                    b = src.in_proj_bias
                    self.q_proj.bias.copy_(b[:E])
                    self.k_proj.bias.copy_(b[E:2*E])
                    self.v_proj.bias.copy_(b[2*E:])
            else:
                self.q_proj.weight.copy_(src.q_proj_weight)
                self.k_proj.weight.copy_(src.k_proj_weight)
                self.v_proj.weight.copy_(src.v_proj_weight)

        self.out_proj = nn.Linear(E, E, bias=src.out_proj.bias is not None)
        with torch.no_grad():
            self.out_proj.weight.copy_(src.out_proj.weight)
            if src.out_proj.bias is not None:
                self.out_proj.bias.copy_(src.out_proj.bias)

        # Copy learnable scale from ScaledMultiheadAttention if present.
        # The saved checkpoints have scale collapsed to 0 (makes attention
        # uniform). Apply the same floor used at training time so exported
        # ONNX doesn't reproduce the bag-of-words failure mode.
        self.scale_param = None
        if hasattr(src, 'scale') and isinstance(src.scale, nn.Parameter):
            floor = float(os.getenv("SCALED_MHA_MIN_SCALE", "1.0"))
            clamped = torch.clamp(src.scale.detach(), min=floor)
            self.scale_param = nn.Parameter(clamped.clone())

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None, **kwargs):
        # Ensure batch-first layout: [B, S, E]
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, S, E = query.shape
        _, T, _ = key.shape
        H = self.num_heads
        D = self.head_dim

        # Apply learnable scale if present
        if self.scale_param is not None:
            query = query * self.scale_param
            key = key * self.scale_param

        # Project Q, K, V
        q = self.q_proj(query).reshape(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
        k = self.k_proj(key).reshape(B, T, H, D).transpose(1, 2)    # [B, H, T, D]
        v = self.v_proj(value).reshape(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, S, T]

        # Apply key padding mask: True = ignore
        if key_padding_mask is not None:
            # [B, T] -> [B, 1, 1, T]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        # Replace NaN from all-masked rows with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, v)  # [B, H, S, D]
        out = out.transpose(1, 2).reshape(B, S, E)  # [B, S, E]
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return out, None


def _replace_mha(module: nn.Module):
    """Recursively replace all MultiheadAttention with ManualMHA."""
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, ManualMHA(child))
        else:
            _replace_mha(child)


def export_all_heads(model_path: str, output_dir: str,
                     d_model=None, nhead=None, num_layers=None, dim_ff=None, cand_feat_dim=None,
                     fixed_shapes=False, fixed_batch=64, fixed_seq=256, fixed_cand=64):
    # Respect per-profile model architecture via env vars — keeps in lockstep
    # with the PyTorch model definitions in py4j_entry_point / mtg_transformer.
    # Defaults match Pauper-Standard (d_model=128, 2 layers); wider profiles
    # like Pauper-Standard-Wide set MODEL_D_MODEL=256, MODEL_NUM_LAYERS=4 via
    # the registry `train_env`.
    if d_model is None:
        d_model = int(os.getenv("MODEL_D_MODEL", "128"))
    if nhead is None:
        nhead = int(os.getenv("MODEL_NHEAD", "4"))
    if num_layers is None:
        num_layers = int(os.getenv("MODEL_NUM_LAYERS", "2"))
    if dim_ff is None:
        dim_ff = int(os.getenv("MODEL_DIM_FF", "512"))
    if cand_feat_dim is None:
        cand_feat_dim = int(os.getenv("MODEL_CAND_FEAT_DIM", "48"))
    os.makedirs(output_dir, exist_ok=True)
    model = MTGTransformerModel(
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_ff, cand_feat_dim=cand_feat_dim,
    )
    model.load(model_path)
    use_fp16 = bool(int(os.getenv("ONNX_EXPORT_FP16", "1")))
    model.cpu().eval()

    if fixed_shapes:
        B, S, N = fixed_batch, fixed_seq, fixed_cand
        print(f"Fixed shapes: batch={B} seq={S} cand={N}")
    else:
        B, S, N = 2, 32, 32

    input_dim = model.input_dim  # respect actual loaded model; input_proj maps input_dim→d_model
    dummy = (
        torch.randn(B, S, input_dim),
        torch.zeros(B, S, dtype=torch.bool),
        torch.zeros(B, S, dtype=torch.long),
        torch.randn(B, N, cand_feat_dim),
        torch.zeros(B, N, dtype=torch.long),
        torch.ones(B, N, dtype=torch.bool),
    )

    if fixed_shapes:
        dynamic_axes = None
    else:
        dynamic_axes = {
            "sequences": {0: "batch", 1: "seq_len"},
            "masks": {0: "batch", 1: "seq_len"},
            "token_ids": {0: "batch", 1: "seq_len"},
            "cand_features": {0: "batch", 1: "max_cand"},
            "cand_ids": {0: "batch", 1: "max_cand"},
            "cand_mask": {0: "batch", 1: "max_cand"},
            "probs": {0: "batch", 1: "max_cand"},
            "value": {0: "batch"},
        }

    for head_id in HEAD_IDS:
        wrapper = SingleHeadScorer(model, head_id)
        wrapper.eval()
        # Replace all MHA modules with ONNX-exportable versions
        _replace_mha(wrapper)

        out_path = os.path.join(output_dir, f"model_{head_id}.onnx")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, out_path,
                opset_version=17,
                input_names=["sequences", "masks", "token_ids",
                             "cand_features", "cand_ids", "cand_mask"],
                output_names=["probs", "value"],
                dynamic_axes=dynamic_axes,
            )
        # Post-export FP16 conversion (cleaner than converting PyTorch model)
        if use_fp16:
            from onnxruntime.transformers.float16 import convert_float_to_float16
            import onnx
            fp32_model = onnx.load(out_path)
            fp16_model = convert_float_to_float16(fp32_model, keep_io_types=True)
            onnx.save(fp16_model, out_path)
        sz = os.path.getsize(out_path)
        print(f"Exported {head_id} -> {out_path} ({sz / 1024:.0f} KB)")

    # Phase 2 belief head: takes only (sequences, masks, token_ids) since
    # archetype prediction is independent of candidate set.
    # Note: we do NOT call _replace_mha here -- the only attention we touch
    # is transformer_layers[*].self_attn which is a ScaledMultiheadAttention
    # (already ONNX-exportable via scaled_dot_product_attention). The cross_attn
    # / cand_self_attn nn.MultiheadAttention modules were already replaced with
    # ManualMHA by earlier action-head exports, but we don't invoke them.
    belief_wrapper = BeliefHeadScorer(model)
    belief_wrapper.eval()
    belief_dummy = (dummy[0], dummy[1], dummy[2])
    if fixed_shapes:
        belief_dynamic = None
    else:
        belief_dynamic = {
            "sequences": {0: "batch", 1: "seq_len"},
            "masks": {0: "batch", 1: "seq_len"},
            "token_ids": {0: "batch", 1: "seq_len"},
            "archetype_logits": {0: "batch"},
        }
    belief_path = os.path.join(output_dir, "model_belief.onnx")
    with torch.no_grad():
        torch.onnx.export(
            belief_wrapper, belief_dummy, belief_path,
            opset_version=17,
            input_names=["sequences", "masks", "token_ids"],
            output_names=["archetype_logits"],
            dynamic_axes=belief_dynamic,
        )
    if use_fp16:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        import onnx
        fp32_model = onnx.load(belief_path)
        fp16_model = convert_float_to_float16(fp32_model, keep_io_types=True)
        onnx.save(fp16_model, belief_path)
    sz = os.path.getsize(belief_path)
    print(f"Exported belief -> {belief_path} ({sz / 1024:.0f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--d-model", type=int, default=int(os.getenv("MODEL_D_MODEL", "128")))
    p.add_argument("--nhead", type=int, default=int(os.getenv("MODEL_NHEAD", "4")))
    p.add_argument("--num-layers", type=int, default=int(os.getenv("MODEL_NUM_LAYERS", "2")))
    p.add_argument("--dim-ff", type=int, default=int(os.getenv("MODEL_DIM_FEEDFORWARD", "512")))
    p.add_argument("--fixed-shapes", action="store_true",
                   help="Export with fixed batch/seq/cand dims for CUDA graph compatibility")
    p.add_argument("--fixed-batch", type=int, default=64)
    p.add_argument("--fixed-seq", type=int, default=256)
    p.add_argument("--fixed-cand", type=int, default=64)
    args = p.parse_args()
    export_all_heads(args.model_path, args.output_dir,
                     d_model=args.d_model, nhead=args.nhead,
                     num_layers=args.num_layers, dim_ff=args.dim_ff,
                     fixed_shapes=args.fixed_shapes,
                     fixed_batch=args.fixed_batch, fixed_seq=args.fixed_seq,
                     fixed_cand=args.fixed_cand)
