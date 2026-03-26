#!/usr/bin/env python3
"""Export MTGTransformerModel to ONNX -- one file per policy head.

Usage:
    py -3.12 MLPythonCode/onnx_export.py --model-path profiles/Pauper-Rally-A/models/model_latest.pt --output-dir /tmp/onnx_test/
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from mtg_transformer import MTGTransformerModel, SingleHeadScorer

HEAD_IDS = ["action", "target", "card_select", "attack", "block"]


def export_all_heads(model_path: str, output_dir: str,
                     d_model=128, nhead=4, num_layers=2, dim_ff=512, cand_feat_dim=48,
                     fixed_shapes=False, fixed_batch=64, fixed_seq=256, fixed_cand=64):
    os.makedirs(output_dir, exist_ok=True)
    model = MTGTransformerModel(
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_ff, cand_feat_dim=cand_feat_dim,
    )
    model.load(model_path)
    model.cpu()
    # Use training mode for export to avoid fused _transformer_encoder_layer_fwd
    # which ONNX can't export. Dropout is 0 so output is identical.
    model.train()
    # Disable dropout explicitly to ensure numerical equivalence
    for layer in model.transformer_layers:
        layer.dropout = torch.nn.Dropout(0.0)
        layer.self_attn.dropout = 0.0

    if fixed_shapes:
        B, S, N = fixed_batch, fixed_seq, fixed_cand
        print(f"Fixed shapes: batch={B} seq={S} cand={N}")
    else:
        B, S, N = 2, 32, 32

    dummy = (
        torch.randn(B, S, d_model),
        torch.zeros(B, S, dtype=torch.bool),
        torch.zeros(B, S, dtype=torch.long),
        torch.randn(B, N, cand_feat_dim),
        torch.zeros(B, N, dtype=torch.long),
        torch.ones(B, N, dtype=torch.bool),
    )

    if fixed_shapes:
        dynamic_shapes = None
    else:
        batch = torch.export.Dim("batch", min=1, max=512)
        seq_len = torch.export.Dim("seq_len", min=1, max=256)
        max_cand = torch.export.Dim("max_cand", min=1, max=512)
        dynamic_shapes = {
            "sequences": {0: batch, 1: seq_len},
            "masks": {0: batch, 1: seq_len},
            "token_ids": {0: batch, 1: seq_len},
            "cand_features": {0: batch, 1: max_cand},
            "cand_ids": {0: batch, 1: max_cand},
            "cand_mask": {0: batch, 1: max_cand},
        }

    for head_id in HEAD_IDS:
        wrapper = SingleHeadScorer(model, head_id)
        wrapper.eval()
        out_path = os.path.join(output_dir, f"model_{head_id}.onnx")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, out_path,
                dynamo=True,
                input_names=["sequences", "masks", "token_ids",
                             "cand_features", "cand_ids", "cand_mask"],
                output_names=["probs", "value"],
                dynamic_shapes=dynamic_shapes,
            )
        print(f"Exported {head_id} -> {out_path}")


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
