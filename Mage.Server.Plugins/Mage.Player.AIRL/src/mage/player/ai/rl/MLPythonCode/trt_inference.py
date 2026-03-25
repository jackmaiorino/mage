"""TensorRT/ONNX Runtime inference context for MTGTransformerModel.

Replaces PyTorch inference with ONNX Runtime sessions (TensorRT EP when available,
falls back to CUDA EP). Handles per-profile model loading, ONNX export on first
use, and re-export on model reload.
"""
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

HEAD_IDS = ["action", "target", "card_select", "attack", "block"]


class TRTInferenceContext:
    """Manages ONNX Runtime sessions for a single profile's model."""

    def __init__(self, profile_id: str, models_dir: str, device: str = "cuda:0"):
        self.profile_id = profile_id
        self.models_dir = Path(models_dir)
        self.onnx_dir = self.models_dir / "onnx"
        self.device = device
        self.device_id = int(device.split(":")[-1]) if ":" in device else 0
        self.sessions: Dict[str, object] = {}
        self._model_mtime: float = 0.0
        self._providers = None

    def _get_providers(self):
        if self._providers is not None:
            return self._providers
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in available:
            cache_dir = str(self.onnx_dir / "trt_cache")
            os.makedirs(cache_dir, exist_ok=True)
            providers.append(("TensorrtExecutionProvider", {
                "device_id": self.device_id,
                "trt_max_workspace_size": 1 << 30,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_dir,
            }))
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": self.device_id}))
        providers.append("CPUExecutionProvider")
        self._providers = providers
        return providers

    def ensure_exported(self, pytorch_model=None) -> bool:
        """Export ONNX models if they don't exist or model_latest.pt is newer. Returns True if exported."""
        model_path = self.models_dir / "model_latest.pt"
        if not model_path.exists():
            return False
        current_mtime = model_path.stat().st_mtime
        onnx_exists = all((self.onnx_dir / f"model_{h}.onnx").exists() for h in HEAD_IDS)
        if onnx_exists and current_mtime <= self._model_mtime:
            return False

        import torch
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from mtg_transformer import MTGTransformerModel, SingleHeadScorer

        if pytorch_model is None:
            d_model = int(os.getenv("MODEL_D_MODEL", "128"))
            nhead = int(os.getenv("MODEL_NHEAD", "4"))
            num_layers = int(os.getenv("MODEL_NUM_LAYERS", "2"))
            dim_ff = int(os.getenv("MODEL_DIM_FEEDFORWARD", "512"))
            pytorch_model = MTGTransformerModel(
                d_model=d_model, nhead=nhead, num_layers=num_layers,
                dim_feedforward=dim_ff, cand_feat_dim=48,
            )
            pytorch_model.load(str(model_path))

        # Clone model to CPU for export -- don't modify the original (learner needs it on GPU)
        import copy
        export_model = copy.deepcopy(pytorch_model).cpu()
        # Training mode to avoid fused TransformerEncoderLayer ops
        export_model.train()
        for layer in export_model.transformer_layers:
            layer.dropout = torch.nn.Dropout(0.0)
            layer.self_attn.dropout = 0.0
        self.onnx_dir.mkdir(parents=True, exist_ok=True)

        B, S, N = 2, 32, 32
        d_model = pytorch_model.d_model
        dummy = (
            torch.randn(B, S, d_model),
            torch.zeros(B, S, dtype=torch.bool),
            torch.zeros(B, S, dtype=torch.long),
            torch.randn(B, N, 48),
            torch.zeros(B, N, dtype=torch.long),
            torch.ones(B, N, dtype=torch.bool),
        )

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

        t0 = time.monotonic()
        for head_id in HEAD_IDS:
            wrapper = SingleHeadScorer(export_model, head_id)
            wrapper.eval()
            out_path = self.onnx_dir / f"model_{head_id}.onnx"
            with torch.no_grad():
                torch.onnx.export(
                    wrapper, dummy, str(out_path),
                    dynamo=True,
                    input_names=["sequences", "masks", "token_ids",
                                 "cand_features", "cand_ids", "cand_mask"],
                    output_names=["probs", "value"],
                    dynamic_shapes=dynamic_shapes,
                )

        elapsed = time.monotonic() - t0
        print(f"[TRT] Exported ONNX for {self.profile_id} in {elapsed:.1f}s", flush=True)
        self._model_mtime = current_mtime
        self.sessions.clear()
        return True

    def _get_session(self, head_id: str):
        if head_id not in self.sessions:
            import onnxruntime as ort
            onnx_path = str(self.onnx_dir / f"model_{head_id}.onnx")
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.log_severity_level = 3
            self.sessions[head_id] = ort.InferenceSession(
                onnx_path, sess_options=so, providers=self._get_providers())
        return self.sessions[head_id]

    def score_batch(self, sequences: np.ndarray, masks: np.ndarray, token_ids: np.ndarray,
                    cand_features: np.ndarray, cand_ids: np.ndarray, cand_mask: np.ndarray,
                    head_id: str = "action") -> Tuple[np.ndarray, np.ndarray]:
        """Run inference via ONNX Runtime. Returns (probs [B,N], value [B,1])."""
        session = self._get_session(head_id)
        probs, value = session.run(None, {
            "sequences": sequences.astype(np.float32),
            "masks": masks.astype(np.bool_),
            "token_ids": token_ids.astype(np.int64),
            "cand_features": cand_features.astype(np.float32),
            "cand_ids": cand_ids.astype(np.int64),
            "cand_mask": cand_mask.astype(np.bool_),
        })
        return probs.astype(np.float32), value.astype(np.float32)

    def reload_if_newer(self) -> bool:
        """Check if model_latest.pt changed and re-export ONNX if needed."""
        model_path = self.models_dir / "model_latest.pt"
        if not model_path.exists():
            return False
        current_mtime = model_path.stat().st_mtime
        if current_mtime <= self._model_mtime:
            return False
        return self.ensure_exported()
