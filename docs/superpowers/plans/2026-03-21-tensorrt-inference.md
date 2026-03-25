# TensorRT Inference Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Python/PyTorch inference with ONNX Runtime + TensorRT to eliminate the 79% Python overhead in the score worker hot path, targeting ~4x throughput improvement.

**Architecture:** Export the MTGTransformerModel.score_candidates() to ONNX (one per policy head), load via ONNX Runtime with TensorRT execution provider. The score worker replaces PyTorch model calls with ort.InferenceSession.run() -- raw numpy in, raw numpy out, no Python tensor construction or GIL contention during inference. Training stays on PyTorch (unchanged).

**Tech Stack:** ONNX, ONNX Runtime with TensorRT EP (`onnxruntime-gpu`), existing PyTorch model for export

---

## Background

### Current inference hot path (Python-bound)
```
Java (bytes) → TCP → gpu_service_host._run_score_batch()
  → gpu_service_core.ProfileContext.score_batch()
    → py4j_entry_point.scoreCandidatesPolicyFlat()
      → np.frombuffer() x6 tensors           ← GIL held (Python)
      → torch.tensor().to(device) x6          ← GIL held (Python+CUDA sync)
      → model.score_candidates()              ← GIL released during CUDA kernels
      → result.cpu().numpy()                  ← GIL held (CUDA sync + copy)
      → np.concatenate().tobytes()            ← GIL held (Python)
  → TCP response
```

**Measured:** Score workers at 100% duty, GPU at 21%, CPU at 59%. Python overhead is 79% of inference time.

### Target inference path (C++ inference, minimal Python)
```
Java (bytes) → TCP → gpu_service_host._run_score_batch()
  → gpu_service_core.ProfileContext.score_batch()
    → trt_inference.TRTInferenceContext.score_batch()
      → np.frombuffer() x6 tensors           ← Still Python but fast (zero-copy views)
      → ort_session.run(numpy_inputs)         ← GIL RELEASED (C++ ONNX Runtime + TensorRT)
      → result.tobytes()                      ← Minimal Python
  → TCP response
```

### Model architecture (for export)
- **Inputs:** 6 tensors (sequences, masks, token_ids, cand_features, cand_ids, cand_mask)
- **Shared encoder:** input_proj → token_id_emb → input_norm → CLS prepend → 2-layer transformer encoder
- **Cross-attention:** candidates ↔ encoded state
- **Self-attention:** among candidates
- **5 head-specific MLPs:** action, target, card_select, attack, block (selected by head_id string)
- **Outputs:** probs [B, N], value [B, 1]
- **Dimensions:** d_model=128, nhead=4, num_layers=2, cand_feat_dim=48

### Export strategy
Export 5 separate ONNX models (one per head). Each includes the full encoder + that head's MLP. The encoder is small (2 layers) so recomputation across heads is negligible vs Python overhead savings. At runtime, select the right ONNX session by head_id.

### Dynamic shapes
- batch_size: varies 1-256 (use ONNX dynamic axis)
- seq_len: bucketed to powers of 2 (8, 16, 32, 64, 128) -- use dynamic axis
- max_candidates: padded per-batch, typically 32 -- use dynamic axis
- d_model=128, cand_feat_dim=48: fixed

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `MLPythonCode/onnx_export.py` | Create | Export MTGTransformerModel to ONNX (one model per head) |
| `MLPythonCode/trt_inference.py` | Create | TRTInferenceContext: load ONNX sessions, run inference, manage per-profile models |
| `MLPythonCode/gpu_service_core.py` | Modify | ProfileContext adds TRT inference path alongside PyTorch |
| `MLPythonCode/py4j_entry_point.py` | Modify | Add ONNX export helper method to PythonEntryPoint |
| `MLPythonCode/mtg_transformer.py` | Modify | Add `export_onnx()` method wrapping score_candidates for single-head export |

---

## Task 1: ONNX Export Script

**Files:**
- Create: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/onnx_export.py`
- Modify: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/mtg_transformer.py`

### Problem
`score_candidates()` uses a string `head_id` parameter to route to different MLPs. ONNX can't handle string-based conditional logic. We need a wrapper that freezes the head selection for export.

- [ ] **Step 1: Add SingleHeadScorer wrapper to mtg_transformer.py**

Add an `nn.Module` wrapper after the `MTGTransformerModel` class that freezes the head_id for ONNX export:

```python
class SingleHeadScorer(nn.Module):
    """Wraps MTGTransformerModel.score_candidates for a single head for ONNX export."""
    def __init__(self, model: MTGTransformerModel, head_id: str):
        super().__init__()
        self.model = model
        self.head_id = head_id

    def forward(self, sequences, masks, token_ids, cand_features, cand_ids, cand_mask):
        probs, value = self.model.score_candidates(
            sequences, masks, token_ids, cand_features, cand_ids, cand_mask,
            head_id=self.head_id)
        return probs, value
```

- [ ] **Step 2: Create onnx_export.py**

```python
"""Export MTGTransformerModel to ONNX -- one file per policy head.

Usage:
    py -3.12 MLPythonCode/onnx_export.py --model-path profiles/Rally-A/models/model_latest.pt --output-dir onnx_models/
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from mtg_transformer import MTGTransformerModel, SingleHeadScorer

HEAD_IDS = ["action", "target", "card_select", "attack", "block"]

def export_all_heads(model_path: str, output_dir: str,
                     d_model=128, nhead=4, num_layers=2, dim_ff=512, cand_feat_dim=48):
    os.makedirs(output_dir, exist_ok=True)
    model = MTGTransformerModel(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                dim_feedforward=dim_ff, cand_feat_dim=cand_feat_dim)
    model.load(model_path)
    model.eval()

    B, S, N = 2, 32, 32  # dummy batch, seq_len, max_candidates
    dummy_inputs = (
        torch.randn(B, S, d_model),          # sequences
        torch.zeros(B, S, dtype=torch.bool),  # masks
        torch.zeros(B, S, dtype=torch.long),  # token_ids
        torch.randn(B, N, cand_feat_dim),     # cand_features
        torch.zeros(B, N, dtype=torch.long),  # cand_ids
        torch.ones(B, N, dtype=torch.bool),   # cand_mask
    )

    for head_id in HEAD_IDS:
        wrapper = SingleHeadScorer(model, head_id)
        wrapper.eval()
        out_path = os.path.join(output_dir, f"model_{head_id}.onnx")
        torch.onnx.export(
            wrapper, dummy_inputs, out_path,
            input_names=["sequences", "masks", "token_ids", "cand_features", "cand_ids", "cand_mask"],
            output_names=["probs", "value"],
            dynamic_axes={
                "sequences": {0: "batch", 1: "seq_len"},
                "masks": {0: "batch", 1: "seq_len"},
                "token_ids": {0: "batch", 1: "seq_len"},
                "cand_features": {0: "batch", 1: "max_cand"},
                "cand_ids": {0: "batch", 1: "max_cand"},
                "cand_mask": {0: "batch", 1: "max_cand"},
                "probs": {0: "batch", 1: "max_cand"},
                "value": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"Exported {head_id} -> {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    export_all_heads(args.model_path, args.output_dir)
```

- [ ] **Step 3: Test ONNX export with an existing model**

```bash
cd Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl
py -3.12 MLPythonCode/onnx_export.py \
  --model-path profiles/Pauper-Rally-A/models/model_latest.pt \
  --output-dir /tmp/onnx_test/
# Expected: 5 .onnx files created (model_action.onnx, model_target.onnx, etc.)
```

- [ ] **Step 4: Validate ONNX output matches PyTorch**

```python
# Quick validation script (run interactively)
import onnxruntime as ort, numpy as np, torch
from mtg_transformer import MTGTransformerModel, SingleHeadScorer

model = MTGTransformerModel(d_model=128, nhead=4, num_layers=2, dim_feedforward=512, cand_feat_dim=48)
model.load("profiles/Pauper-Rally-A/models/model_latest.pt")
model.eval()

B, S, N = 4, 32, 32
seqs = torch.randn(B, S, 128)
masks = torch.zeros(B, S, dtype=torch.bool)
toks = torch.zeros(B, S, dtype=torch.long)
cfeat = torch.randn(B, N, 48)
cids = torch.zeros(B, N, dtype=torch.long)
cmask = torch.ones(B, N, dtype=torch.bool)

# PyTorch reference
with torch.no_grad():
    pt_probs, pt_val = model.score_candidates(seqs, masks, toks, cfeat, cids, cmask, head_id="action")

# ONNX
sess = ort.InferenceSession("/tmp/onnx_test/model_action.onnx", providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {
    "sequences": seqs.numpy(), "masks": masks.numpy(), "token_ids": toks.numpy(),
    "cand_features": cfeat.numpy(), "cand_ids": cids.numpy(), "cand_mask": cmask.numpy(),
})
print(f"Max diff probs: {np.max(np.abs(pt_probs.numpy() - ort_out[0]))}")
print(f"Max diff value: {np.max(np.abs(pt_val.numpy() - ort_out[1]))}")
# Expected: Max diff < 1e-5
```

- [ ] **Step 5: Commit**

---

## Task 2: TRT Inference Context

**Files:**
- Create: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/trt_inference.py`

- [ ] **Step 1: Create trt_inference.py**

```python
"""TensorRT/ONNX Runtime inference context for MTGTransformerModel.

Replaces PyTorch inference with ONNX Runtime sessions (TensorRT EP when available).
Handles per-profile model loading, ONNX export on first use, and re-export on model reload.
"""
import os, time, numpy as np
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
        self.sessions: Dict[str, "ort.InferenceSession"] = {}
        self._model_mtime: float = 0.0
        self._providers = self._select_providers()

    def _select_providers(self):
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in available:
            providers.append(("TensorrtExecutionProvider", {
                "device_id": self.device_id,
                "trt_max_workspace_size": 1 << 30,  # 1GB
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(self.onnx_dir / "trt_cache"),
            }))
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": self.device_id}))
        providers.append("CPUExecutionProvider")
        return providers

    def ensure_exported(self, pytorch_model=None) -> bool:
        """Export ONNX models if they don't exist or are stale. Returns True if (re-)exported."""
        model_path = self.models_dir / "model_latest.pt"
        if not model_path.exists():
            return False
        current_mtime = model_path.stat().st_mtime
        onnx_exists = all((self.onnx_dir / f"model_{h}.onnx").exists() for h in HEAD_IDS)
        if onnx_exists and current_mtime <= self._model_mtime:
            return False

        # Need to export
        import torch
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

        pytorch_model.eval()
        pytorch_model.cpu()
        self.onnx_dir.mkdir(parents=True, exist_ok=True)

        B, S, N = 2, 32, 32
        d_model = pytorch_model.d_model
        cand_feat_dim = 48
        dummy = (
            torch.randn(B, S, d_model),
            torch.zeros(B, S, dtype=torch.bool),
            torch.zeros(B, S, dtype=torch.long),
            torch.randn(B, N, cand_feat_dim),
            torch.zeros(B, N, dtype=torch.long),
            torch.ones(B, N, dtype=torch.bool),
        )

        for head_id in HEAD_IDS:
            wrapper = SingleHeadScorer(pytorch_model, head_id)
            wrapper.eval()
            out_path = self.onnx_dir / f"model_{head_id}.onnx"
            with torch.no_grad():
                torch.onnx.export(
                    wrapper, dummy, str(out_path),
                    input_names=["sequences", "masks", "token_ids",
                                 "cand_features", "cand_ids", "cand_mask"],
                    output_names=["probs", "value"],
                    dynamic_axes={
                        "sequences": {0: "batch", 1: "seq_len"},
                        "masks": {0: "batch", 1: "seq_len"},
                        "token_ids": {0: "batch", 1: "seq_len"},
                        "cand_features": {0: "batch", 1: "max_cand"},
                        "cand_ids": {0: "batch", 1: "max_cand"},
                        "cand_mask": {0: "batch", 1: "max_cand"},
                        "probs": {0: "batch", 1: "max_cand"},
                        "value": {0: "batch"},
                    },
                    opset_version=17,
                    do_constant_folding=True,
                )

        self._model_mtime = current_mtime
        self.sessions.clear()  # Force reload
        return True

    def _get_session(self, head_id: str) -> "ort.InferenceSession":
        if head_id not in self.sessions:
            import onnxruntime as ort
            onnx_path = str(self.onnx_dir / f"model_{head_id}.onnx")
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sessions[head_id] = ort.InferenceSession(
                onnx_path, sess_options=sess_opts, providers=self._providers)
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
        return probs, value

    def reload_if_newer(self) -> bool:
        """Check if model_latest.pt changed and re-export ONNX if so."""
        model_path = self.models_dir / "model_latest.pt"
        if not model_path.exists():
            return False
        current_mtime = model_path.stat().st_mtime
        if current_mtime <= self._model_mtime:
            return False
        self.ensure_exported()
        return True
```

- [ ] **Step 2: Commit**

---

## Task 3: Integrate into GPU Service Score Path

**Files:**
- Modify: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_core.py`

The goal: when `USE_TRT_INFERENCE=1` env var is set, the ProfileContext's inference path uses `TRTInferenceContext` instead of `PythonEntryPoint.scoreCandidatesPolicyFlat()`. Training stays on PyTorch.

- [ ] **Step 1: Add TRT inference path to ProfileContext**

In `ProfileContext.__init__()`, after creating the PythonEntryPoint, optionally create a TRTInferenceContext:

```python
# After self.entry = PythonEntryPoint() ...
self._trt_ctx = None
if os.getenv("USE_TRT_INFERENCE", "0") == "1" and self.role == "inference":
    from trt_inference import TRTInferenceContext
    models_dir = self.entry.model_dir  # path to profile's models/ dir
    self._trt_ctx = TRTInferenceContext(profile_id, models_dir, cuda_device or "cuda:0")
    self._trt_ctx.ensure_exported(self.entry.model)
```

- [ ] **Step 2: Add TRT score_batch method to ProfileContext**

Add a new method that bypasses PyTorch:

```python
def score_batch_trt(self, sequences_bytes, masks_bytes, token_ids_bytes,
                    cand_feat_bytes, cand_ids_bytes, cand_mask_bytes,
                    batch_size, seq_len, d_model, max_candidates, cand_feat_dim,
                    head_id="action") -> bytes:
    """Fast inference via ONNX Runtime / TensorRT. No PyTorch, minimal GIL."""
    seq = np.frombuffer(sequences_bytes, dtype='<f4').reshape(batch_size, seq_len, d_model)
    mask = np.frombuffer(masks_bytes, dtype='<i4').reshape(batch_size, seq_len).astype(np.bool_)
    tok = np.frombuffer(token_ids_bytes, dtype='<i4').reshape(batch_size, seq_len).astype(np.int64)
    cfeat = np.frombuffer(cand_feat_bytes, dtype='<f4').reshape(batch_size, max_candidates, cand_feat_dim)
    cids = np.frombuffer(cand_ids_bytes, dtype='<i4').reshape(batch_size, max_candidates).astype(np.int64)
    cmask = np.frombuffer(cand_mask_bytes, dtype='<i4').reshape(batch_size, max_candidates).astype(np.bool_)

    probs, value = self._trt_ctx.score_batch(seq, mask, tok, cfeat, cids, cmask, head_id)

    out = np.concatenate((probs.astype(np.float32), value.astype(np.float32)), axis=1)
    return out.tobytes()
```

- [ ] **Step 3: Route score_batch to TRT when available**

Modify the existing `score_batch()` method to check for TRT context:

```python
def score_batch(self, ...):
    if self._trt_ctx is not None:
        return self.score_batch_trt(...)
    # ... existing PyTorch path
```

- [ ] **Step 4: Handle model reload for TRT**

Modify `reload_latest_model_if_newer()` to also trigger ONNX re-export:

```python
def reload_latest_model_if_newer(self, path=None):
    with self.lock:
        reloaded = bool(self.entry.reloadLatestModelIfNewer(path))
        if reloaded and self._trt_ctx is not None:
            self._trt_ctx.ensure_exported(self.entry.model)
        return reloaded
```

- [ ] **Step 5: Commit**

---

## Task 4: Install Dependencies and Benchmark

**Files:**
- Modify: `scripts/run_local_pbt.py` (add USE_TRT_INFERENCE env var)

- [ ] **Step 1: Install onnxruntime-gpu**

```bash
py -3.12 -m pip install onnxruntime-gpu onnx
```

Note: `onnxruntime-gpu` includes TensorRT EP on Windows if TensorRT is installed. If TensorRT is not available, it falls back to CUDA EP (still faster than PyTorch due to graph optimization).

- [ ] **Step 2: Add USE_TRT_INFERENCE to local PBT**

In `run_local_pbt.py`, in `start_gpu_service()`:
```python
env["USE_TRT_INFERENCE"] = os.getenv("USE_TRT_INFERENCE", "1")
```

- [ ] **Step 3: Run benchmark -- PyTorch baseline**

```bash
cd C:/Users/Jack/IdeaProjects/mage
TRAIN_PROFILES=5 USE_TRT_INFERENCE=0 py -3.12 scripts/run_local_pbt.py &
# Wait 2 min warmup, measure:
sleep 120
S1=$(curl -s http://localhost:27100/metrics | grep train_batches_total | grep -v '#' | awk '{print $2}')
sleep 60
S2=$(curl -s http://localhost:27100/metrics | grep train_batches_total | grep -v '#' | awk '{print $2}')
echo "PyTorch: $((S2-S1)) eps/min"
```

- [ ] **Step 4: Run benchmark -- TRT inference**

```bash
TRAIN_PROFILES=5 USE_TRT_INFERENCE=1 py -3.12 scripts/run_local_pbt.py &
# Wait 3 min (first run does ONNX export + TRT engine build)
sleep 180
S1=$(curl -s http://localhost:27100/metrics | grep train_batches_total | grep -v '#' | awk '{print $2}')
sleep 60
S2=$(curl -s http://localhost:27100/metrics | grep train_batches_total | grep -v '#' | awk '{print $2}')
echo "TRT: $((S2-S1)) eps/min"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
wmic cpu get LoadPercentage | grep -o '[0-9]*'
```

Expected: 2-4x improvement in eps/min. GPU utilization should increase significantly. CPU should become the new bottleneck.

- [ ] **Step 5: Commit**

---

## Task 5: HPC Compatibility

**Files:**
- Modify: `scripts/hpc/gpu_head.sbatch`
- Modify: `scripts/hpc/cpu_worker.sh`

- [ ] **Step 1: Add USE_TRT_INFERENCE to gpu_head.sbatch**

```bash
export USE_TRT_INFERENCE="${USE_TRT_INFERENCE:-1}"
```

- [ ] **Step 2: Ensure onnxruntime-gpu is in the HPC Python environment**

The HPC runtime bundle includes Python packages. Add `onnxruntime-gpu` and `onnx` to the bundle's pip requirements or install at runtime.

- [ ] **Step 3: Test on HPC**

Submit a short test run and verify ONNX export + TRT inference works on H100/A100.

- [ ] **Step 4: Commit**

---

## Risks and Fallbacks

1. **ONNX export fails for custom ops:** The `ScaledMultiheadAttention` in mtg_transformer.py has a learnable scale parameter. If ONNX can't export it, replace with standard `nn.MultiheadAttention` + manual scaling in the export wrapper.

2. **TensorRT EP not available on Windows:** Falls back to CUDA EP (ONNX Runtime's own CUDA kernels). Still faster than PyTorch due to graph optimization and fused kernels. No TensorRT installation needed.

3. **Dynamic shapes cause TRT engine recompilation:** Each new (batch_size, seq_len, max_candidates) combination triggers TRT engine rebuild (~30s). Mitigate by bucketing inputs to common shape profiles (already done for seq_len).

4. **Model reload during training:** The `ensure_exported()` check on mtime handles this. Re-export takes ~2s for the small model. During re-export, the old ONNX sessions remain valid until explicitly replaced.

5. **Numerical precision:** TRT with FP16 may produce slightly different outputs. Validate with the comparison script in Task 1 Step 4. If precision matters, disable `trt_fp16_enable`.
