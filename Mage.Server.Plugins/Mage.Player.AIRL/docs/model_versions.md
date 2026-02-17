# Model Versions

## v2.1 (current) — 2-layer, d_model=128, cross-attention
- ~2.1M parameters (added cross-attention for candidate scoring)
- Estimated saturation: 50k-200k episodes (10-20M decision steps)
- VRAM: ~30MB (train), ~10MB (inference)
- Architecture: Candidates cross-attend to full state sequence, then concat with CLS for MLP scoring
- Critic head still uses CLS-only (unchanged)
- Designed for: single-GPU training at 1-2 eps/sec

## v2 — 2-layer, d_model=128
- ~2M parameters
- CLS-concat candidate scoring (no cross-attention)
- VRAM: ~30MB (train), ~10MB (inference)
- Designed for: single-GPU training at 1-2 eps/sec

## v1 (archived) — 6-layer, d_model=512
- ~55M parameters
- Estimated saturation: 20M+ episodes
- VRAM: ~800MB (train), ~200MB (inference)
- Impractical on single consumer GPU at current throughput
