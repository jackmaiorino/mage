# Java ONNX Runtime Inference Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Python-based GPU inference with ONNX Runtime Java bindings for ~10-20x lower inference latency (46ms → 2-5ms), eliminating the Python GIL bottleneck and enabling higher training throughput.

**Architecture:** Create `OnnxPythonModel` implementing the existing `PythonModel` interface. It loads ONNX models (one per policy head, pre-exported by `onnx_export.py`) via ONNX Runtime Java GPU provider. Inference runs in-process in the JVM -- no TCP, no Python, no serialization. Training stays on the Python GPU service (unchanged). The model factory selects between SharedGpuPythonModel (for training) and OnnxPythonModel (for inference-only eval or when `PY_SERVICE_MODE=onnx`).

**Tech Stack:** ONNX Runtime Java GPU (`com.microsoft.onnxruntime:onnxruntime_gpu`), existing ONNX export (`onnx_export.py`), existing `PythonModel` interface

---

## Background

### Current inference path (Python-bound, 46ms latency)
```
Java game thread → TCP socket → Python GIL → numpy/torch → GPU → numpy → Python → TCP → Java
```

### Target inference path (in-process, ~2-5ms latency)
```
Java game thread → ORT Java API → GPU → ORT Java API → Java game thread
```

### Key constraint
Training MUST stay on Python/PyTorch (optimizer state, GAE, PPO loss computation). Only inference moves to Java. The OnnxPythonModel delegates all training methods (enqueueTraining, trainMulligan, etc.) to the existing SharedGpuPythonModel or no-ops them.

### Data shapes (from existing codebase)
- **Inputs:** sequences [1, seqLen, 128] float32, masks [1, seqLen] bool, token_ids [1, seqLen] int64, cand_features [1, maxCand, 48] float32, cand_ids [1, maxCand] int64, cand_mask [1, maxCand] bool
- **Outputs:** probs [1, maxCand] float32, value [1, 1] float32
- **5 head models:** action, target, card_select, attack, block
- **ONNX files:** `profiles/<PROFILE>/models/onnx/model_<head>.onnx`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `Mage.Server.Plugins/Mage.Player.AIRL/pom.xml` | Modify | Add onnxruntime_gpu Maven dependency |
| `...rl/OnnxInferenceModel.java` | Create | PythonModel impl using ONNX Runtime Java sessions |
| `...rl/PythonModelFactory.java` | Modify | Add `onnx` mode that creates OnnxInferenceModel |
| `...rl/SharedGpuPythonModel.java` | Modify | Extract padding/bucketing utils as static (shared with OnnxInferenceModel) |

---

## Task 1: Add Maven Dependency

**Files:**
- Modify: `Mage.Server.Plugins/Mage.Player.AIRL/pom.xml`

- [ ] **Step 1: Add ONNX Runtime GPU dependency**

Add to `<dependencies>` section:
```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime_gpu</artifactId>
    <version>1.17.1</version>
</dependency>
```

- [ ] **Step 2: Verify compilation**

```bash
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

- [ ] **Step 3: Verify ONNX Runtime loads**

```bash
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java \
  "-Dexec.mainClass=ai.onnxruntime.OrtEnvironment" 2>&1 | head -5
```
Or add a quick test in Java:
```java
OrtEnvironment env = OrtEnvironment.getEnvironment();
System.out.println("ONNX Runtime version: " + env.toString());
```

- [ ] **Step 4: Commit**

---

## Task 2: Create OnnxInferenceModel

**Files:**
- Create: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/OnnxInferenceModel.java`

This is the core class. It implements `PythonModel` and uses ONNX Runtime Java for inference.

- [ ] **Step 1: Create OnnxInferenceModel class**

Key design:
- Loads 5 ONNX sessions (one per head) on construction
- `scoreCandidates()` converts Java arrays to OnnxTensors, calls session.run(), extracts results
- Training methods delegate to a wrapped `SharedGpuPythonModel` (if training needed) or no-op
- Thread-safe: ORT sessions are thread-safe for concurrent reads
- Reloads ONNX files when model_latest.pt changes (re-export needed)

```java
public final class OnnxInferenceModel implements PythonModel {
    private final OrtEnvironment env;
    private final Map<String, OrtSession> sessions;  // headId -> session
    private final String modelsDir;
    private PythonModel trainingDelegate;  // SharedGpuPythonModel for training ops

    // Constructor: load ONNX models from profiles/<PROFILE>/models/onnx/
    // scoreCandidates: no batching needed, single inference per call
    // Input tensors: create OnnxTensor from flat float[]/long[] arrays
    // Output: extract float[] probs + float value from OrtSession.Result
}
```

Key implementation details:

**Input tensor creation (per call, no batching):**
```java
// sequences: float[1][seqLen][128] from SequenceOutput.tokens
float[][][] seqTensor = new float[1][seqLen][128];
System.arraycopy(state.getSequence(), 0, seqTensor[0], 0, seqLen);

// masks: boolean[1][seqLen] from SequenceOutput.mask (convert int[] to boolean[])
boolean[][] maskTensor = new boolean[1][seqLen];
for (int i = 0; i < seqLen; i++) maskTensor[0][i] = state.getMask()[i] != 0;

// token_ids: long[1][seqLen] from SequenceOutput.tokenIds (convert int[] to long[])
long[][] tokTensor = new long[1][seqLen];
for (int i = 0; i < seqLen; i++) tokTensor[0][i] = state.getTokenIds()[i];

// cand_features: float[1][maxCand][48] from candidateFeatures
// cand_ids: long[1][maxCand] from candidateActionIds (int[] to long[])
// cand_mask: boolean[1][maxCand] from candidateMask (int[] to boolean[])
```

**Session.run:**
```java
Map<String, OnnxTensor> inputs = Map.of(
    "sequences", OnnxTensor.createTensor(env, seqTensor),
    "masks", OnnxTensor.createTensor(env, maskTensor),
    "token_ids", OnnxTensor.createTensor(env, tokTensor),
    "cand_features", OnnxTensor.createTensor(env, candFeatTensor),
    "cand_ids", OnnxTensor.createTensor(env, candIdTensor),
    "cand_mask", OnnxTensor.createTensor(env, candMaskTensor)
);
OrtSession.Result result = sessions.get(headId).run(inputs);
float[][] probs = (float[][]) result.get("probs").get().getValue();
float[][] value = (float[][]) result.get("value").get().getValue();
return new PredictionResult(probs[0], value[0][0]);
```

**Important: close OnnxTensors after use to prevent native memory leaks.**

- [ ] **Step 2: Implement all PythonModel interface methods**

For training methods (enqueueTraining, trainMulligan, etc.):
- If `trainingDelegate != null`, forward the call
- Otherwise, no-op (inference-only mode)

For mulligan prediction:
- Use a separate ONNX session for the mulligan model, OR
- Delegate to trainingDelegate for now (mulligan model is small, Python overhead is acceptable)

- [ ] **Step 3: Add model reload support**

```java
public void reloadIfNewer() {
    // Check model_latest.pt mtime
    // If changed, re-export ONNX via subprocess call to onnx_export.py
    // Then reload sessions
}
```

- [ ] **Step 4: Commit**

---

## Task 3: Wire Into PythonModelFactory

**Files:**
- Modify: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/PythonModelFactory.java`

- [ ] **Step 1: Add onnx mode**

In `getInstance()`:
```java
case "onnx":
case "onnx_gpu":
    return new OnnxInferenceModel(profileDir, deviceId);
```

- [ ] **Step 2: Add hybrid mode (ONNX inference + Python training)**

For training, we still need the Python GPU service for:
- PPO training (forward + backward + optimizer step)
- Mulligan training
- Model saving

Create a `HybridModel` that wraps both:
```java
case "hybrid":
    OnnxInferenceModel onnx = new OnnxInferenceModel(profileDir, deviceId);
    SharedGpuPythonModel gpu = new SharedGpuPythonModel();
    onnx.setTrainingDelegate(gpu);
    return onnx;
```

The game runners call `scoreCandidates()` → hits ONNX (fast, in-process).
Training calls (`enqueueTraining`, etc.) → forwarded to SharedGpuPythonModel → TCP → Python.

- [ ] **Step 3: Update run_local_pbt.py to use hybrid mode**

```python
env["PY_SERVICE_MODE"] = os.getenv("PY_SERVICE_MODE", "hybrid")
```

- [ ] **Step 4: Commit**

---

## Task 4: Benchmark

- [ ] **Step 1: Export ONNX models for Pauper-Standard**

```bash
cd Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl
py -3.12 MLPythonCode/onnx_export.py --model-path profiles/Pauper-Standard/models/model_latest.pt --output-dir profiles/Pauper-Standard/models/onnx/
```

- [ ] **Step 2: Baseline measurement (Python inference)**

```bash
PY_SERVICE_MODE=shared_gpu py -3.12 scripts/run_local_pbt.py &
# Wait 2 min, measure eps/min
```

- [ ] **Step 3: ONNX measurement**

```bash
PY_SERVICE_MODE=hybrid py -3.12 scripts/run_local_pbt.py &
# Wait 2 min, measure eps/min
# Check CPU% (should be much higher since inference is ~10x faster)
```

- [ ] **Step 4: Commit and document results**

Expected results:
- Inference latency: 46ms → 2-5ms
- CPU utilization: 34% → 80%+
- Throughput: 90 eps/min → 200+ eps/min
- GPU (nvidia-smi): 20% → 5% (even less compute needed without Python overhead)

---

## Task 5: Model Reload During Training

**Files:**
- Modify: `OnnxInferenceModel.java`

The Python GPU service trains the model and saves `model_latest.pt`. The ONNX model needs to be re-exported and reloaded periodically.

- [ ] **Step 1: Add periodic re-export check**

Every N seconds (e.g., 60s), check if `model_latest.pt` is newer than the ONNX files.
If so, call `onnx_export.py` as a subprocess to re-export, then reload sessions.

- [ ] **Step 2: Handle concurrent access during reload**

Use read-write lock: inference uses read lock, reload uses write lock.
During reload (~2s), inference falls back to the old sessions until new ones are ready.

- [ ] **Step 3: Commit**

---

## Risks and Fallbacks

1. **ONNX Runtime Java GPU provider not available on Windows:** Falls back to CPU provider. Still faster than Python due to no TCP/serialization, but no GPU acceleration. Install CUDA toolkit if needed.

2. **ONNX model output differs from PyTorch:** We validated ~0.02 max diff in probs during the TensorRT experiment. Acceptable for RL training.

3. **Native memory leaks from OnnxTensor:** Each `OnnxTensor.createTensor()` allocates native memory. Must call `.close()` on all tensors and results after use. Use try-with-resources.

4. **Model reload during training:** The re-export takes ~2s. During this time, inference uses stale model. This is the same as the current Python model publish/reload latency.

5. **Mulligan model not in ONNX:** Mulligan inference stays on Python for now. It's called once per game (not per decision), so the overhead is negligible (~1ms per game vs ~100 inference calls per game at 46ms each).

6. **Maven dependency conflicts:** `onnxruntime_gpu` bundles native libraries. If conflicts arise with existing CUDA/cuDNN, use `onnxruntime` (CPU-only) as fallback.
