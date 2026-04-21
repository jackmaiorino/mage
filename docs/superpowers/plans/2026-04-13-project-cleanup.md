# Project Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ~22GB of dead code, stale logs, and abandoned experiments while documenting what was tried and why.

**Architecture:** Delete dead files, clean stale profiles, write a project timeline, update CLAUDE.md. No code logic changes -- pure cleanup.

**Tech Stack:** git, bash, markdown

---

## File Map

| Action | Path | Size | Reason |
|--------|------|------|--------|
| Delete | `scripts/rust-infer-server/` | 1.9 GB | Abandoned; Java ONNX replaced it |
| Delete | `rl/logs/` | 12 GB | Stale March logs, no active use |
| Delete | `MLPythonCode/mtg_ai.log` | 118 MB | March training artifact |
| Delete | `MLPythonCode/mulligan_training.log` | 141 MB | March training artifact |
| Delete | `MLPythonCode/mulligan_trace.jsonl` | 422 MB | March training artifact |
| Delete | `MLPythonCode/trt_inference.py` | 7.6 KB | Superseded by Java ONNX |
| Delete | `MLPythonCode/onnx_export_int8.py` | 1.3 KB | INT8 non-viable on GPU |
| Delete | `profiles/{dead profiles}/` | ~7 GB | Not in registry, stale |
| Delete | `profiles/*/onnx/{int8,fp32,fp16_backup,int8_from_fp32}/` | ~1.3 GB | Investigation artifacts |
| Delete | `scripts/deploy-budget-gpu.bat` | 11 KB | AWS relic |
| Delete | `scripts/teardown-budget-gpu.bat` | 4.8 KB | AWS relic |
| Delete | `scripts/run-training-oneliner.ps1` | 9 KB | Superseded by run_local_pbt.py |
| Delete | `scripts/rl-benchmark-before-after.ps1` | 686 B | Ad-hoc, unused |
| Delete | `scripts/hpc/_tmp_relaunch.sh` | - | Temp debug script |
| Delete | `scripts/hpc/_tmp_resubmit.sh` | - | Temp debug script |
| Delete | `scripts/hpc/test_gpu.sh` | 47 B | Trivial test |
| Create | `docs/PROJECT_TIMELINE.md` | - | What was tried and why |
| Clean | `rl/league/pauper_spy_pbt_registry.json` | - | Remove entries for deleted profiles |

---

### Task 1: Write project timeline

**Files:**
- Create: `docs/PROJECT_TIMELINE.md`

- [ ] **Step 1: Create the timeline document**

```markdown
# XMage RL Training -- Project Timeline

## January 2026: Foundation
- Built RL agent (ComputerPlayerRL) with PPO training
- PythonMLBridge (Py4J) for Java-Python communication
- Initial training on Pauper Spy Combo deck
- AWS budget GPU scripts (deploy-budget-gpu.bat) -- abandoned, too expensive

## February 2026: League & Benchmarking
- League evaluation framework (rl-league-eval.ps1, rl-league-run.ps1)
- Benchmarking suite (bench-* profiles) -- measured baseline performance
- Multiple deck profiles created for testing

## March 2026: HPC Multi-Node Architecture
- Zaratan HPC cluster integration (gpu_head.sbatch, cpu_worker.sh, add_satellites.sh)
- GPU head + CPU satellite architecture with split-device mode
- SharedGpuPythonModel: TCP-based batched inference protocol
- Spy Combo PBT training at scale (profiles A/B/C)
- Score worker auto-scaling, round-robin fix
- **Explored:** Rust ONNX inference server (scripts/rust-infer-server/) -- built but
  never deployed. Python GIL was the bottleneck, not language speed.
- **Explored:** TensorRT Python inference (trt_inference.py) -- worked but
  superseded by the Java ONNX approach which eliminated Python from inference entirely.
- Peak HPC throughput: 5.7 eps/sec with 10 satellites

## Late March 2026: Java ONNX Runtime
- OnnxInferenceModel.java: in-process ONNX Runtime with CUDA EP
- Eliminated Python from inference path (was GIL-bottlenecked)
- FP16 ONNX export via ManualMHA decomposition (PyTorch can't export nn.MultiheadAttention)
- Hybrid mode (PY_SERVICE_MODE=hybrid): Java ONNX for inference, Python for training
- 6.5x throughput improvement over Python inference

## April 2026 (early): Multi-Deck PBT & Critical Bug Fixes
- Switched from Spy Combo to multi-deck Pauper pool: Wildfire, Rally, Affinity, Elves
- **Fixed:** Per-profile deck mismatch -- all profiles were playing random decks
- **Fixed:** One-sided training -- opponent side wasn't generating training data
- **Fixed:** Mulligan Q-learning collapse -- replaced with REINFORCE, then merged into main model
- Dual-sided training doubled effective throughput
- Eval checkpoints vs CP7 (heuristic opponent) for absolute strength measurement

## April 2026 (mid): Throughput Investigation
- Problem: ONNX (5.5GB) + PyTorch training (6GB) on 12GB GPU = 91% OOM rate
- **Tested:** INT8 ONNX quantization -- 10-30x slower, ORT dequantizes to FP32. Dead end.
- **Tested:** Model shrinking -- already at minimum (d_model=128, 2 layers). Dead end.
- **Tested:** Remote GPU training (Haley's RTX 4060) -- infrastructure works, PC intermittently available.
- **Analyzed:** Time-multiplex (pause ONNX during training) -- viable but medium effort.
- **Found:** ONNX_GPU_MEM_LIMIT_MB wasn't being passed to JVM (defaulted to 5120MB instead of 2048MB).
- **Fixed:** Capped ONNX arena to 2048MB + added torch.cuda.empty_cache() after training batches.
- Result: 1.28 -> 10.8 rows/sec (8.5x improvement), OOM rate 91% -> 30%.
- **Recommended:** Second local GPU ($50-80 used GTX 1070). PCIe x16 Gen4 slot confirmed free.

## Current Architecture (April 2026)
- **Inference:** Java ONNX Runtime (CUDA EP, FP16, batched) -- OnnxInferenceModel.java
- **Training:** Python PyTorch GPU service (PPO) -- gpu_service_host.py
- **Orchestrator:** run_local_pbt.py -- 4 profiles, PBT exploitation, eval checkpoints
- **Model:** d_model=128, 2 layers, 4 heads, 17.6M params (95% embeddings)
- **Throughput:** ~10.8 rows/sec total, ~2.7/profile, 30% OOM rate
- **Next:** Second GPU eliminates OOMs entirely
```

- [ ] **Step 2: Commit timeline**

```bash
git add docs/PROJECT_TIMELINE.md
git commit -m "docs: add project timeline summarizing what was tried and why"
```

---

### Task 2: Delete dead code files

**Files:**
- Delete: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/trt_inference.py`
- Delete: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/onnx_export_int8.py`
- Modify: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_core.py` (remove trt_inference import)

- [ ] **Step 1: Remove the TRT import guard in gpu_service_core.py**

Find and remove the conditional import of trt_inference (around line 79-83):

```python
# REMOVE this block:
        if os.getenv("USE_TRT_INFERENCE", "0") == "1" and self.role == "inference":
            ...
            from trt_inference import TRTInferenceContext, HEAD_IDS
```

Replace with nothing (or a pass if needed for the if/else structure).

- [ ] **Step 2: Delete dead Python files**

```bash
cd C:/Users/Jack/IdeaProjects/mage
rm Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/trt_inference.py
rm Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/onnx_export_int8.py
```

- [ ] **Step 3: Verify no other references**

```bash
grep -r "trt_inference\|onnx_export_int8" Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/ --include="*.py"
```

Expected: no matches (or only in the file we already edited).

- [ ] **Step 4: Commit**

```bash
git add -A Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/
git commit -m "chore: remove dead trt_inference.py and onnx_export_int8.py"
```

---

### Task 3: Delete Rust inference server

**Files:**
- Delete: `scripts/rust-infer-server/` (1.9 GB)

- [ ] **Step 1: Verify no references**

```bash
grep -r "rust-infer\|mage-infer\|rust.infer" scripts/ Mage.Server.Plugins/ --include="*.py" --include="*.java" --include="*.sh" --include="*.ps1" --include="*.bat" 2>/dev/null
```

Expected: no matches.

- [ ] **Step 2: Delete**

```bash
rm -rf scripts/rust-infer-server/
```

- [ ] **Step 3: Commit**

```bash
git add scripts/rust-infer-server/
git commit -m "chore: remove abandoned Rust inference server (1.9GB)

Superseded by Java ONNX Runtime (OnnxInferenceModel.java).
See docs/PROJECT_TIMELINE.md for context."
```

---

### Task 4: Delete dead scripts

**Files:**
- Delete: `scripts/deploy-budget-gpu.bat`
- Delete: `scripts/teardown-budget-gpu.bat`
- Delete: `scripts/run-training-oneliner.ps1`
- Delete: `scripts/rl-benchmark-before-after.ps1`
- Delete: `scripts/hpc/_tmp_relaunch.sh`
- Delete: `scripts/hpc/_tmp_resubmit.sh`
- Delete: `scripts/hpc/test_gpu.sh`

- [ ] **Step 1: Delete all dead scripts**

```bash
cd C:/Users/Jack/IdeaProjects/mage
rm -f scripts/deploy-budget-gpu.bat
rm -f scripts/teardown-budget-gpu.bat
rm -f scripts/run-training-oneliner.ps1
rm -f scripts/rl-benchmark-before-after.ps1
rm -f scripts/hpc/_tmp_relaunch.sh
rm -f scripts/hpc/_tmp_resubmit.sh
rm -f scripts/hpc/test_gpu.sh
```

- [ ] **Step 2: Commit**

```bash
git add -A scripts/
git commit -m "chore: remove dead scripts (AWS, one-liner, temp debug)"
```

---

### Task 5: Delete stale logs (not in git -- just disk cleanup)

These are untracked runtime artifacts. No git commit needed.

**Files:**
- Delete: `rl/logs/` (12 GB)
- Delete: `MLPythonCode/mtg_ai.log` (118 MB)
- Delete: `MLPythonCode/mulligan_training.log` (141 MB)
- Delete: `MLPythonCode/mulligan_trace.jsonl` (422 MB)
- Delete: `MLPythonCode/VRAM_diagnostics.log`

- [ ] **Step 1: Delete stale logs**

```bash
cd C:/Users/Jack/IdeaProjects/mage/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl
rm -rf logs/
rm -f MLPythonCode/mtg_ai.log
rm -f MLPythonCode/mulligan_training.log
rm -f MLPythonCode/mulligan_trace.jsonl
rm -f MLPythonCode/VRAM_diagnostics.log
```

- [ ] **Step 2: Verify removal**

```bash
du -sh Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/logs/ 2>/dev/null || echo "logs/ removed"
```

Expected: "logs/ removed"

---

### Task 6: Delete dead profile directories (not in git -- disk cleanup)

Profiles NOT in the PBT registry and not actively training. Preserves the 4 active profiles
(Pauper-Wildfire, Pauper-Rally, Pauper-Affinity, Pauper-Elves) plus registry entries
(A/B variants, Spy-Combo-A/B).

**Files to delete:**

```
profiles/Pauper-MonoRedRally/          4.3 GB  (not in registry)
profiles/rally-local/                  2.1 GB  (not in registry)
profiles/Pauper-Standard/              6.7 GB  (registry: active=False)
profiles/engine-bench/                 291 MB  (benchmark)
profiles/smoke-pad-test/               267 MB  (test)
profiles/Vintage-Cube/                 96 MB   (not in registry)
profiles/Pauper-Spy-Combo/             67 MB   (not in registry, A/B are)
profiles/Pauper-Spy-Combo-C/           67 MB   (not in registry)
profiles/Pauper-Caw-Gates/             16 KB   (empty stub)
profiles/Pauper-Grixis-Affinity/       16 KB   (empty stub)
profiles/Pauper-Jund-Wildfire/         16 KB   (empty stub)
profiles/Pauper-Mono-Blue-Faeries/     12 KB   (empty stub)
profiles/Pauper-Mono-Blue-Terror/      12 KB   (empty stub)
profiles/Pauper-Mono-Red-Burn/         8 KB    (empty stub)
profiles/Pauper-Mono-Red-Rally/        8 KB    (empty stub)
profiles/bench-*/                      ~3 MB   (all benchmark variants)
profiles/noop-test/                    25 KB   (test)
profiles/profileA/                     3 KB    (test)
profiles/profileB/                     3 KB    (test)
profiles/test/                         0       (empty)
```

- [ ] **Step 1: Delete dead profiles**

```bash
cd C:/Users/Jack/IdeaProjects/mage/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles

# Large dead profiles
rm -rf Pauper-MonoRedRally rally-local Pauper-Standard engine-bench smoke-pad-test Vintage-Cube
rm -rf Pauper-Spy-Combo Pauper-Spy-Combo-C

# Empty stubs
rm -rf Pauper-Caw-Gates Pauper-Grixis-Affinity Pauper-Jund-Wildfire
rm -rf Pauper-Mono-Blue-Faeries Pauper-Mono-Blue-Terror Pauper-Mono-Red-Burn Pauper-Mono-Red-Rally

# Benchmark/test profiles
rm -rf bench-all-opt bench-all-opt-cp7 bench-base-200 bench-baseline bench-final bench-final-200
rm -rf bench-noclone bench-noclone-sp bench-noclone-test bench-profile bench-rl-opp bench-selfplay
rm -rf bench-sim bench-simonly-200
rm -rf noop-test profileA profileB test
```

- [ ] **Step 2: Verify only active profiles remain**

```bash
ls -d */ | sort
```

Expected (13 directories):
```
Pauper-Affinity/
Pauper-Affinity-A/
Pauper-Affinity-B/
Pauper-Elves/
Pauper-Elves-A/
Pauper-Rally/
Pauper-Rally-A/
Pauper-Rally-B/
Pauper-Spy-Combo-A/
Pauper-Spy-Combo-B/
Pauper-Wildfire/
Pauper-Wildfire-A/
```

(Plus Pauper-Elves-B and Pauper-Wildfire-B if they exist as directories.)

---

### Task 7: Clean investigation artifacts from active profiles

**Files to delete from each of the 4 active profiles:**

- `profiles/*/models/onnx/int8/`
- `profiles/*/models/onnx/int8_from_fp32/`
- `profiles/*/models/onnx/fp32/`
- `profiles/*/models/onnx/fp16_backup/`

- [ ] **Step 1: Delete investigation artifacts**

```bash
cd C:/Users/Jack/IdeaProjects/mage/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles
for p in Pauper-Rally Pauper-Wildfire Pauper-Affinity Pauper-Elves; do
  rm -rf "$p/models/onnx/int8"
  rm -rf "$p/models/onnx/int8_from_fp32"
  rm -rf "$p/models/onnx/fp32"
  rm -rf "$p/models/onnx/fp16_backup"
  echo "Cleaned $p"
done
```

---

### Task 8: Clean PBT registry

**Files:**
- Modify: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json`

- [ ] **Step 1: Remove entries for deleted profiles**

Remove any registry entries whose profile directories no longer exist (Pauper-Standard, any Wildfire-B/Elves-B that don't have directories). Keep all entries that have matching profile directories.

- [ ] **Step 2: Commit registry cleanup**

```bash
git add Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json
git commit -m "chore: remove registry entries for deleted profiles"
```

---

### Task 9: Clean local-training artifacts

Not in git. Disk cleanup only.

- [ ] **Step 1: Remove stale test logs**

```bash
cd C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt
rm -f config_a.log config_c.log trainer_overnight.log sync.log sync_models.log
```

Keep: `gpu_service.log`, `trainer.log`, `config_tuned.log` (active run), `throughput_investigation_2026-04-12.md`, `eval_history.csv`, `winrate_charts.png`.

---

### Task 10: Update stale design docs

**Files:**
- Modify: `docs/superpowers/plans/2026-03-21-tensorrt-inference.md` (add superseded note)

- [ ] **Step 1: Add superseded header to TensorRT plan**

Add to the top of the file:

```markdown
> **SUPERSEDED** by [Java ONNX Runtime](2026-03-25-java-onnx-inference.md). TensorRT Python inference was replaced by in-process Java ONNX Runtime for lower latency and no GIL contention. See [Project Timeline](../../PROJECT_TIMELINE.md).
```

- [ ] **Step 2: Commit**

```bash
git add docs/
git commit -m "docs: mark TensorRT plan as superseded"
```

---

### Task 11: Final verification

- [ ] **Step 1: Check disk savings**

```bash
cd C:/Users/Jack/IdeaProjects/mage
du -sh .
du -sh Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/
```

Expected: ~22 GB freed.

- [ ] **Step 2: Verify training still works**

```bash
# Training should still be running from the tuned config
tasklist | grep -iE "java|python3"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Expected: java.exe and python3.12.exe running, GPU active.

- [ ] **Step 3: Verify compilation**

```bash
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

Expected: clean compile (the deleted trt_inference.py reference was only in Python, not Java).
