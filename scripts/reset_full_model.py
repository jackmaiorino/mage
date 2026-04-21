"""Hard reset: fresh init the whole model except vocabulary layers.

The encoder was trained under ScaledMHA.scale=0 (uniform/broken attention)
for 37k+ episodes. Q/K projections got no useful gradient; V projections
learned features that only make sense under uniform attention. Cross-attn
and candidate self-attn were trained on those broken features too.
Retraining from this starting point is harder than from scratch because
the weights are tuned for a regime (scale=0) different from the one we
now run (scale=1).

This script preserves ONLY the vocabulary / raw-feature layers:
- token_id_emb   : card identity embeddings (nn.Embedding)
- input_proj     : raw feature -> d_model projection (nn.Linear)
- cand_feat_proj : candidate feature MLP (does not depend on encoder)
- action_id_emb  : action identity embeddings

Everything else gets fresh xavier/standard init, and the optimizer state is
cleared.

Run: py -3.12 scripts/reset_full_model.py
Optional: PATCH_PROFILES="Pauper-Wildfire,Pauper-Rally,..."
"""
import os
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_ROOT = REPO_ROOT / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "profiles"

# Prefixes to KEEP (preserve learned vocab/raw-feature mappings)
PRESERVE_PREFIXES = (
    "token_id_emb.",
    "action_id_emb.",
    "input_proj.",
    "cand_feat_proj.",
)

# Known scalar keys (shape [1] or []) and their target values
SCALAR_RESETS = {
    "input_scale": 1.0,
    "value_scale": 1.0,
    "temperature": 0.7,
    # attention scales -- all reset to 1.0 (matching SCALED_MHA_MIN_SCALE floor)
}


def classify(key: str, tensor: torch.Tensor) -> str:
    if tensor.numel() == 1:
        return "scalar"
    if "self_attn.scale" in key and tensor.numel() == 1:
        return "scalar"
    if key.endswith(".weight"):
        return "layernorm_weight" if tensor.dim() == 1 else "linear_weight"
    if key.endswith(".bias"):
        return "layernorm_bias" if tensor.dim() == 1 else "linear_bias"
    # Parameters like cls_token / critic_cls_token (3D)
    if tensor.dim() >= 2:
        return "linear_weight"
    return "layernorm_bias"


def fresh_init(key: str, tensor: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "scalar":
        target = None
        # Match on exact key or suffix
        for k_suffix, v in SCALAR_RESETS.items():
            if key == k_suffix or key.endswith("." + k_suffix):
                target = v
                break
        if target is None:
            # e.g., self_attn.scale -> 1.0
            target = 1.0
        out = torch.empty_like(tensor)
        out.fill_(float(target))
        return out
    if kind == "layernorm_weight":
        return torch.ones_like(tensor)
    if kind == "layernorm_bias":
        return torch.zeros_like(tensor)
    if kind == "linear_bias":
        return torch.zeros_like(tensor)
    if kind == "linear_weight":
        out = torch.empty_like(tensor)
        if out.dim() >= 2:
            nn.init.xavier_uniform_(out, gain=0.5)
        else:
            # 1D linear weight rare (e.g., LayerNorm slipping through) -> ones
            out.fill_(1.0)
        return out
    return tensor


def should_preserve(key: str) -> bool:
    return any(key.startswith(p) for p in PRESERVE_PREFIXES)


def patch_checkpoint(pt_path: Path) -> bool:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        print(f"  SKIP {pt_path}: unexpected format")
        return False
    sd = ckpt["state_dict"]
    preserved = 0
    reset_count = 0
    for k in list(sd.keys()):
        v = sd[k]
        if should_preserve(k):
            preserved += 1
            continue
        kind = classify(k, v)
        sd[k] = fresh_init(k, v, kind)
        reset_count += 1
    # Clear Adam state -- momentum from old (broken) gradients
    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        ckpt["optimizer_state_dict"] = None
    # Also reset training counters so schedules restart
    for ctr_key in ("train_step_counter", "main_train_sample_counter", "gae_enabled_step"):
        if ctr_key in ckpt:
            ckpt[ctr_key] = 0 if ctr_key != "gae_enabled_step" else None
    backup = pt_path.with_suffix(pt_path.suffix + ".prehard_reset")
    if not backup.exists():
        pt_path.rename(backup)
        torch.save(ckpt, pt_path)
        print(f"  {pt_path.name}: preserved {preserved}, reset {reset_count}, backup {backup.name}")
    else:
        torch.save(ckpt, pt_path)
        print(f"  {pt_path.name}: preserved {preserved}, reset {reset_count} (backup existed)")
    return True


def main() -> None:
    filter_csv = os.getenv("PATCH_PROFILES", "")
    only = {x.strip() for x in filter_csv.split(",") if x.strip()}
    patched = 0
    for ckpt in sorted(PROFILES_ROOT.glob("*/models/model_latest.pt")):
        profile = ckpt.parents[1].name
        if only and profile not in only:
            continue
        print(f"[{profile}]")
        if patch_checkpoint(ckpt):
            patched += 1
        model_pt = ckpt.parent / "model.pt"
        if model_pt.exists():
            if patch_checkpoint(model_pt):
                patched += 1
    print(f"\nTotal files patched: {patched}")


if __name__ == "__main__":
    main()
