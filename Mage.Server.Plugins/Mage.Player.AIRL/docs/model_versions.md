# Model Versions

## v2 (current) — 2-layer, d_model=128
- ~2M parameters
- Estimated saturation: 50k-200k episodes (10-20M decision steps)
- VRAM: ~30MB (train), ~10MB (inference)
- Designed for: single-GPU training at 1-2 eps/sec

## v1 (archived) — 6-layer, d_model=512
- ~55M parameters
- Estimated saturation: 20M+ episodes
- VRAM: ~800MB (train), ~200MB (inference)
- Impractical on single consumer GPU at current throughput
