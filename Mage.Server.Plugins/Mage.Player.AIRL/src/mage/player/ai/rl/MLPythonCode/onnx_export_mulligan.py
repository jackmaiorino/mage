#!/usr/bin/env python3
"""Export MulliganNet to ONNX.

Usage:
    py -3.12 onnx_export_mulligan.py --model-path profiles/Pauper-Rally/models/mulligan_model.pt --output-dir profiles/Pauper-Rally/models/onnx/
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from mulligan_model import MulliganNet

FEATURE_DIM = 71  # 1 + 3 + 7 + 60


def export_mulligan(model_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    model = MulliganNet(vocab_size=65536, embed_dim=32, max_hand=7, max_deck=60)
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.cpu()
    model.train()  # avoid fused ops that ONNX can't export

    # Build dummy input with valid card IDs (embedding indices must be in range)
    # Format: [mulligan_num(1), land_count(1), creature_count(1), avg_cmc(1), hand_ids(7), deck_ids(60)]
    dummy = torch.zeros(2, FEATURE_DIM)
    dummy[:, 0] = 1.0   # mulligan_num
    dummy[:, 1] = 3.0   # land_count
    dummy[:, 2] = 2.0   # creature_count
    dummy[:, 3] = 2.5   # avg_cmc
    # hand_ids and deck_ids: use small valid indices (1-100)
    dummy[:, 4:11] = torch.randint(1, 100, (2, 7)).float()
    dummy[:, 11:71] = torch.randint(1, 100, (2, 60)).float()
    out_path = os.path.join(output_dir, "mulligan_model.onnx")

    torch.onnx.export(
        model, (dummy,),
        out_path,
        input_names=["features"],
        output_names=["logit"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported mulligan -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export_mulligan(args.model_path, args.output_dir)
