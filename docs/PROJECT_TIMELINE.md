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
