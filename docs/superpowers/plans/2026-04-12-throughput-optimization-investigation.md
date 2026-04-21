# Throughput Optimization Investigation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine the optimal local training configuration by empirically testing all viable throughput optimizations and analyzing architectural changes.

**Architecture:** Single RTX 4070 Super (12GB) running ONNX inference + PyTorch training. Current bottleneck: VRAM contention between ONNX (~5GB) and PyTorch (~5-6GB) on a 12GB GPU causes 27-47% OOM rate and limits throughput to ~2.5 rows/sec per profile vs theoretical ~4+ rows/sec. Model: d_model=512, 6 layers, 8 heads, 5 ONNX heads at 34MB each (FP16).

**Tech Stack:** Java 8+ / ONNX Runtime 1.19 (CUDA EP), Python 3.12 / PyTorch (CUDA), RTX 4070 Super 12GB local, RTX 4060 8GB remote (Haley's PC, intermittently available).

---

## Current State

- **ONNX inference:** FP16, local GPU, 5.3GB VRAM, 35% GPU util, ~48ms avg latency (8-9ms actual run, rest is batch assembly wait)
- **Training:** Broken -- pointing at Haley's PC (10.0.0.22:26100) which is offline. Games complete but no weight updates.
- **Profiles:** 4 (Rally, Wildfire, Affinity, Elves), 64 runners total
- **Best historical:** ~7 effective eps/sec (1.75 rows/sec/profile x4, dual-sided) with FP16 ONNX + GPU training at CUDA_MEM_FRACTION=0.55, but 27% OOM rate

## Measurement Protocol

For each config, after 2-minute warmup:
1. Record CSV row counts across all 4 profiles
2. Wait 10 minutes
3. Record CSV row counts again
4. Compute: `rows_delta / 600s = rows/sec per profile`
5. Capture: `nvidia-smi` (GPU util, VRAM), OOM count from gpu_service.log, train batch success/latency from TRAIN_DIAG, ONNX flush stats from trainer.log

Save results to `local-training/local_pbt/throughput_investigation_2026-04-12.md`.

---

### Task 1: Kill Current Run and Establish Clean State

**Files:**
- None modified

- [ ] **Step 1: Stop all training processes**

```bash
taskkill //F //IM java.exe
taskkill //F //IM python3.12.exe
```

- [ ] **Step 2: Verify GPU is clear**

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

Expected: 0%, ~0 MiB used (or just desktop compositor)

- [ ] **Step 3: Record current CSV row counts as baseline**

```bash
for p in Pauper-Rally Pauper-Wildfire Pauper-Affinity Pauper-Elves; do
  f="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/$p/logs/stats/training_stats.csv"
  wc -l < "$f"
done
```

---

### Task 2: Config A -- Baseline (FP16 ONNX GPU + PyTorch GPU Training, Local)

This is the known best config before the remote experiment. Re-establish it as the reference point.

**Files:**
- Modify: `scripts/run_local_pbt.py` (revert GPU_SERVICE_ENDPOINT to localhost, set env vars)

- [ ] **Step 1: Configure for local-only training**

In `scripts/run_local_pbt.py`, the `start_gpu_service` method already defaults to localhost. Ensure no `GPU_SERVICE_ENDPOINT` env var is set externally:

```bash
unset GPU_SERVICE_ENDPOINT
```

Key env vars for this config (already defaults in run_local_pbt.py):
- `PY_SERVICE_MODE=hybrid` (ONNX inference + Python training)
- `TRAIN_CUDA_DEVICE=cuda:0` (PyTorch trains on GPU)
- `CUDA_MEM_FRACTION=0.55` (PyTorch gets 6.6GB cap)
- `ONNX_GPU_MEM_LIMIT_MB=3072` (nominal, ORT grows beyond this)
- `ONNX_EXPORT_FP16=1` (half-size ONNX models)
- `NUM_GAME_RUNNERS=64`

- [ ] **Step 2: Compile and start training**

```bash
cd C:/Users/Jack/IdeaProjects/mage
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
py -3.12 scripts/run_local_pbt.py > local-training/local_pbt/config_a.log 2>&1 &
```

- [ ] **Step 3: Wait for warmup (ONNX load + first games)**

Wait ~2 minutes. Verify ONNX loaded and games are starting:
```bash
grep -a "ONNX.*flush\|PROFILE\|rows/sec" local-training/local_pbt/trainer.log | tail -5
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Expected: VRAM ~10-11GB (ONNX 5GB + PyTorch 5-6GB), GPU util >30%

- [ ] **Step 4: Measure for 10 minutes**

Record start CSV counts, wait 600s, record end counts. Also capture:
- OOM count: `grep -ac "OutOfMemoryError\|CUDA out of memory\|oom_count" local-training/local_pbt/gpu_service.log`
- Train batch stats: `grep -a "TRAIN_DIAG" local-training/local_pbt/gpu_service.log | tail -10`
- ONNX stats: `grep -a "ONNX.*flush" local-training/local_pbt/trainer.log | tail -5`
- GPU: `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader`

- [ ] **Step 5: Record Config A results**

```
Config A: FP16 ONNX GPU + PyTorch GPU (CUDA_MEM_FRACTION=0.55)
  rows/sec/profile: ___
  total rows/sec:   ___
  OOM count:        ___
  OOM rate:         ___%
  GPU util:         ___%
  VRAM:             ___ / 12282 MiB
  Infer latency:    ___ms avg
  Train batch:      ___ms avg, ___% success
```

- [ ] **Step 6: Stop training**

```bash
taskkill //F //IM java.exe
taskkill //F //IM python3.12.exe
```

---

### Task 3: Config B -- FP16 ONNX GPU + Remote PyTorch Training (Haley's RTX 4060)

**Depends on:** Haley's PC (10.0.0.22) being online. If offline, skip and note in results.

**Files:**
- None modified (env var override only)

- [ ] **Step 1: Check if Haley's PC is reachable**

```bash
ping -n 1 10.0.0.22
ssh haley@10.0.0.22 "echo OK"
```

If unreachable, skip this task entirely and record "SKIPPED: host offline" in results.

- [ ] **Step 2: Start GPU service on remote**

```bash
ssh haley@10.0.0.22 "cd C:\\Users\\haley\\mage && py -3.12 Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_host.py" &
```

Or use schtasks if SSH session management is needed. Verify it's listening:
```bash
ssh haley@10.0.0.22 "curl -s http://localhost:27100/metrics | head -5"
```

- [ ] **Step 3: Start local training pointing at remote**

```bash
GPU_SERVICE_ENDPOINT=10.0.0.22:26100 py -3.12 scripts/run_local_pbt.py > local-training/local_pbt/config_b.log 2>&1 &
```

ONNX inference stays local (full 12GB available, no contention).
Training data flows over LAN to Haley's 4060.

- [ ] **Step 4: Measure for 10 minutes**

Same measurement protocol as Config A. Additionally check:
- Remote GPU: `ssh haley@10.0.0.22 "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"`
- Network errors: `grep -ac "request failed\|Connection refused\|timed out" local-training/local_pbt/trainer.log`

- [ ] **Step 5: Record Config B results**

```
Config B: FP16 ONNX GPU + Remote PyTorch GPU (4060 over LAN)
  rows/sec/profile: ___
  total rows/sec:   ___
  OOM count:        ___ (local), ___ (remote)
  GPU util:         ___% (local), ___% (remote)
  VRAM:             ___ / 12282 MiB (local), ___ / 8188 MiB (remote)
  Network errors:   ___
  Train batch:      ___ms avg
```

- [ ] **Step 6: Stop training (local and remote)**

---

### Task 4: Export INT8 ONNX Models

**Files:**
- Create: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/onnx_export_int8.py`

- [ ] **Step 1: Write INT8 quantization script**

Uses ONNX Runtime's dynamic quantization (no calibration data needed):

```python
#!/usr/bin/env python3
"""Quantize FP16 ONNX models to INT8 using ONNX Runtime dynamic quantization."""
import argparse
import sys
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

HEAD_IDS = ["action", "target", "card_select", "attack", "block"]

def quantize_all(onnx_dir: str, output_dir: str = None):
    onnx_path = Path(onnx_dir)
    out_path = Path(output_dir) if output_dir else onnx_path / "int8"
    out_path.mkdir(parents=True, exist_ok=True)

    for head in HEAD_IDS:
        src = onnx_path / f"model_{head}.onnx"
        dst = out_path / f"model_{head}.onnx"
        if not src.exists():
            print(f"SKIP {head}: {src} not found")
            continue
        print(f"Quantizing {head}: {src} -> {dst}")
        quantize_dynamic(
            str(src), str(dst),
            weight_type=QuantType.QInt8,
            optimize_model=True,
        )
        src_mb = src.stat().st_size / 1024 / 1024
        dst_mb = dst.stat().st_size / 1024 / 1024
        print(f"  {src_mb:.1f}MB -> {dst_mb:.1f}MB ({dst_mb/src_mb*100:.0f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    quantize_all(args.onnx_dir, args.output_dir)
```

- [ ] **Step 2: Run quantization on all profiles**

```bash
for p in Pauper-Rally Pauper-Wildfire Pauper-Affinity Pauper-Elves; do
  onnx_dir="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/$p/models/onnx"
  py -3.12 Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/onnx_export_int8.py \
    --onnx-dir "$onnx_dir"
done
```

Expected output: each 34MB FP16 model -> ~17MB INT8 model (50% reduction).

- [ ] **Step 3: Verify INT8 models produce reasonable output**

Quick smoke test -- load one model and check outputs aren't garbage:

```python
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession("path/to/int8/model_action.onnx", providers=["CPUExecutionProvider"])
# Create dummy input matching model shape
# Check output logits are finite and not all-zero
```

---

### Task 5: Config C -- INT8 ONNX GPU + PyTorch GPU Training (Local)

**Files:**
- Modify: `scripts/run_local_pbt.py` (point ONNX to int8 subdir) OR
- Modify: `OnnxInferenceModel.java` (add INT8 path override) OR
- Env var: symlink/copy int8 models over FP16 ones

Simplest approach: temporarily copy INT8 models over the FP16 ones in the onnx/ dirs.

- [ ] **Step 1: Swap in INT8 models**

```bash
for p in Pauper-Rally Pauper-Wildfire Pauper-Affinity Pauper-Elves; do
  onnx_dir="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/$p/models/onnx"
  # Back up FP16
  mkdir -p "$onnx_dir/fp16_backup"
  cp "$onnx_dir"/model_*.onnx "$onnx_dir/fp16_backup/"
  # Copy INT8 over
  cp "$onnx_dir/int8/"model_*.onnx "$onnx_dir/"
done
```

- [ ] **Step 2: Start training with INT8 ONNX + GPU training**

Key change: with INT8 ONNX using ~2-3GB instead of 5GB, try higher CUDA_MEM_FRACTION:

```bash
unset GPU_SERVICE_ENDPOINT
CUDA_MEM_FRACTION=0.70 py -3.12 scripts/run_local_pbt.py > local-training/local_pbt/config_c.log 2>&1 &
```

- [ ] **Step 3: Measure for 10 minutes**

Same protocol. Key metrics to watch:
- VRAM usage (should be significantly lower with INT8)
- OOM rate (should be lower with more headroom)
- Inference accuracy (watch for degraded game quality -- winrate collapse)

- [ ] **Step 4: Record Config C results**

```
Config C: INT8 ONNX GPU + PyTorch GPU (CUDA_MEM_FRACTION=0.70)
  rows/sec/profile: ___
  total rows/sec:   ___
  OOM count:        ___
  OOM rate:         ___%
  GPU util:         ___%
  VRAM:             ___ / 12282 MiB
  ONNX VRAM:        ~___ MiB (vs ~5000 MiB FP16)
  Infer latency:    ___ms avg
  Train batch:      ___ms avg, ___% success
  Notes on accuracy: ___
```

- [ ] **Step 5: Restore FP16 models, stop training**

```bash
for p in Pauper-Rally Pauper-Wildfire Pauper-Affinity Pauper-Elves; do
  onnx_dir="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/$p/models/onnx"
  cp "$onnx_dir/fp16_backup/"model_*.onnx "$onnx_dir/"
done
taskkill //F //IM java.exe; taskkill //F //IM python3.12.exe
```

---

### Task 6: Config D -- INT8 ONNX GPU + Remote PyTorch Training

**Depends on:** Haley's PC online AND Config C showing INT8 ONNX works correctly.

- [ ] **Step 1: Swap in INT8 models (same as Task 5 Step 1)**

- [ ] **Step 2: Start with remote training endpoint**

```bash
GPU_SERVICE_ENDPOINT=10.0.0.22:26100 py -3.12 scripts/run_local_pbt.py > local-training/local_pbt/config_d.log 2>&1 &
```

- [ ] **Step 3: Measure for 10 minutes, record results**

```
Config D: INT8 ONNX GPU + Remote PyTorch GPU (4060)
  rows/sec/profile: ___
  total rows/sec:   ___
  OOM count:        ___ (should be 0)
  GPU util:         ___% (local), ___% (remote)
  VRAM:             ___ / 12282 MiB (local)
  Infer latency:    ___ms
  Train batch:      ___ms
```

- [ ] **Step 4: Restore FP16, stop everything**

---

### Task 7: Time-Multiplex Analysis and Prototype

No empirical test -- this is a design analysis and feasibility assessment.

**Files to analyze:**
- `OnnxInferenceModel.java` (inference gating mechanism)
- `SharedGpuPythonModel.java` (training trigger)
- `gpu_service_host.py` (training batch execution)

- [ ] **Step 1: Analyze the coordination requirement**

The idea: during PyTorch training batches (~1-3s on GPU), pause ONNX inference. Game runners block on their CompletableFuture (they already do this). After training completes, resume ONNX. The GPU alternates between inference and training with no VRAM contention.

Key questions to answer:
1. How often do training batches fire? (Every N episodes, measure from logs)
2. How long does each training batch take on GPU? (From TRAIN_DIAG logs)
3. What fraction of time would inference be paused? (batch_duration * batch_frequency)
4. What's the net throughput: faster training (GPU vs CPU) minus inference pauses?

- [ ] **Step 2: Calculate time-multiplex math**

From existing logs, extract:
```bash
# Training batch frequency and duration
grep -a "TRAIN_DIAG\|train_batch\|batch_duration" local-training/local_pbt/gpu_service.log | tail -20
# Episodes between batches
grep -a "batch" local-training/local_pbt/gpu_service.log | tail -20
```

Math:
- Current (no time-mux): ONNX runs 100% of time, PyTorch trains on CPU at ~10s/batch
- Time-mux: ONNX paused for ~1-3s per batch, PyTorch trains on GPU at ~0.3-1s/batch
- If training fires every 30s and takes 1s on GPU: 1/30 = 3.3% inference downtime
- vs CPU training at 10s: training 3x faster, inference only 3.3% slower
- Net: significantly higher throughput

- [ ] **Step 3: Design the coordination protocol**

Option A (simplest): **Volatile boolean gate in OnnxInferenceModel**
- Python GPU service sends HTTP POST to a tiny Java HTTP endpoint before/after training
- OnnxInferenceModel.flushQueue() checks `volatile boolean trainingInProgress`
- If true, sleep 100ms and retry (game runners block on their future)
- Pros: ~50 lines of code, no protocol changes
- Cons: 100ms polling granularity, HTTP overhead

Option B: **TCP opcode extension**
- Add opcode TRAIN_GATE_ACQUIRE / TRAIN_GATE_RELEASE to SharedGpuPythonModel protocol
- Python sends gate signals over existing TCP connection
- OnnxInferenceModel holds a ReentrantReadWriteLock: inference takes read lock, training takes write lock
- Pros: precise, no polling, no new HTTP server
- Cons: protocol change, need to route signal from SharedGpuPythonModel to OnnxInferenceModel

Option C: **No gate, just VRAM sizing**
- Don't coordinate at all
- Use INT8 ONNX (~2-3GB) + careful PyTorch VRAM cap
- If total peak < 12GB, no OOMs even without coordination
- This is effectively what Config C tests

- [ ] **Step 4: Estimate implementation effort**

- Option A: ~2 hours (HTTP server in Java, signal in Python, boolean gate in ONNX flush)
- Option B: ~4 hours (protocol extension, lock plumbing)
- Option C: ~0 hours (just Config C from Task 5)
- Recommend: Try Config C first. If OOM rate is still >10%, implement Option A.

- [ ] **Step 5: Record analysis results**

Document: training frequency, batch duration, time-mux math, recommended approach, estimated effort.

---

### Task 8: Model Architecture Analysis

Analyze whether d_model=512 is oversized for the task and what shrinking would gain.

- [ ] **Step 1: Profile current model compute**

```bash
# Check parameter count
py -3.12 -c "
import torch, sys
sys.path.insert(0, 'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode')
from mtg_transformer import MTGTransformerModel
m = MTGTransformerModel()
total = sum(p.numel() for p in m.parameters())
print(f'Total params: {total:,}')
print(f'd_model=512: {total:,} params, ~{total*4/1024/1024:.1f}MB FP32, ~{total*2/1024/1024:.1f}MB FP16')
m2 = MTGTransformerModel(d_model=256, dim_feedforward=1024, nhead=4, num_layers=4)
total2 = sum(p.numel() for p in m2.parameters())
print(f'd_model=256: {total2:,} params, ~{total2*4/1024/1024:.1f}MB FP32, ~{total2*2/1024/1024:.1f}MB FP16')
print(f'Reduction: {(1-total2/total)*100:.0f}%')
"
```

- [ ] **Step 2: Profile ONNX inference time by model size**

Export a d_model=256 model to ONNX and benchmark inference:

```python
# Create small model, export to ONNX, benchmark vs current
import time, torch, onnxruntime as ort
# ... (export small model)
# Run 100 inferences, compare latency
```

- [ ] **Step 3: Assess capacity vs speed tradeoff**

Questions to answer:
- Is d_model=512 overkill for Pauper MTG decisions? (4 simple decks, limited card pool)
- What's the inference speedup from 512->256? (2-4x expected from attention scaling)
- What's the VRAM savings? (roughly 75% less model VRAM)
- Would retraining from scratch be needed? (yes -- can't prune a trained model easily)
- How long would retraining take to reach current performance? (unknown, ~24-48h based on episode rates)

- [ ] **Step 4: Record analysis results**

Document: param counts, ONNX sizes, inference benchmarks, capacity assessment, recommendation.

---

### Task 9: Hardware Path Assessment

Quick check on whether a second local GPU is viable.

- [ ] **Step 1: Check PCIe slot availability**

```bash
wmic path win32_systemslot get SlotDesignation,CurrentUsage,MaxDataWidth 2>/dev/null
# Or just note: user was told to "open the case and look"
```

- [ ] **Step 2: Document hardware option**

- Does the motherboard have a second PCIe x16 slot?
- PSU wattage sufficient for two GPUs?
- Cost: used GTX 1070/1080 = $50-80 for training-only GPU
- This permanently solves VRAM contention for ~$70

---

### Task 10: Compile Results and Recommend

- [ ] **Step 1: Create results document**

Save to `local-training/local_pbt/throughput_investigation_2026-04-12.md`:

```markdown
# Throughput Optimization Investigation Results (2026-04-12)

## Hardware
- Local: RTX 4070 Super (12GB), 24 cores, 32GB RAM
- Remote: RTX 4060 (8GB), 16 cores (Haley's PC, intermittent)

## Results

| Config | rows/sec/profile | total rows/sec | OOM rate | GPU util | VRAM |
|--------|-----------------|---------------|----------|----------|------|
| A: FP16 + local GPU train | ___ | ___ | ___% | ___% | ___ MiB |
| B: FP16 + remote train | ___ | ___ | ___% | ___% | ___ MiB |
| C: INT8 + local GPU train | ___ | ___ | ___% | ___% | ___ MiB |
| D: INT8 + remote train | ___ | ___ | ___% | ___% | ___ MiB |

## Analysis: Time-Multiplex
[results from Task 7]

## Analysis: Model Shrinking
[results from Task 8]

## Analysis: Second GPU
[results from Task 9]

## Recommendation
[winner config + rationale + next steps]
```

- [ ] **Step 2: Update memory with findings**

Save optimal config to memory for future sessions.
