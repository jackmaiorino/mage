"""Reset critic head weights for shared-encoder refactor.

After merging the critic encoder into the actor encoder, the critic head's
weights (critic_norm/norm1/proj1/proj2/value_scale) are calibrated for inputs
coming from the OLD separate critic_transformer. They don't interpret the
shared CLS correctly. Quickest recovery: reset to fresh init.

Run: py -3.12 scripts/reset_critic_head.py
"""
import os
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_ROOT = REPO_ROOT / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "profiles"

# Critic head keys to reset (the MLP on top of shared CLS)
CRITIC_HEAD_PREFIXES = (
    "critic_norm.",
    "critic_norm1.",
    "critic_proj1.",
    "critic_proj2.",
)
CRITIC_HEAD_SCALARS = ("value_scale",)


def fresh_init_like(tensor: torch.Tensor, kind: str) -> torch.Tensor:
    """Return a fresh random tensor with sensible init for the given key kind."""
    if kind == "layernorm_weight":
        return torch.ones_like(tensor)
    if kind == "layernorm_bias":
        return torch.zeros_like(tensor)
    if kind == "linear_bias":
        return torch.zeros_like(tensor)
    if kind == "linear_weight":
        out = torch.empty_like(tensor)
        nn.init.xavier_uniform_(out, gain=0.5)
        return out
    if kind == "value_scale":
        return torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device).view_as(tensor)
    return tensor


def classify_key(key: str, tensor: torch.Tensor) -> str:
    if key in CRITIC_HEAD_SCALARS:
        return "value_scale"
    if key.startswith("critic_norm.") or key.startswith("critic_norm1."):
        return "layernorm_weight" if key.endswith(".weight") else "layernorm_bias"
    if key.startswith(("critic_proj1.", "critic_proj2.")):
        return "linear_weight" if key.endswith(".weight") else "linear_bias"
    return "unknown"


def patch_checkpoint(pt_path: Path) -> bool:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        print(f"  SKIP {pt_path}: unexpected format")
        return False
    sd = ckpt["state_dict"]
    changed = []
    for k in list(sd.keys()):
        v = sd[k]
        if any(k.startswith(p) for p in CRITIC_HEAD_PREFIXES) or k in CRITIC_HEAD_SCALARS:
            kind = classify_key(k, v)
            if kind == "unknown":
                continue
            sd[k] = fresh_init_like(v, kind)
            changed.append((k, kind))
    if not changed:
        print(f"  {pt_path.name}: no critic-head keys found")
        return False
    # Drop optimizer state: Adam's momentum is calibrated for the OLD head.
    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        ckpt["optimizer_state_dict"] = None
        changed.append(("optimizer_state_dict", "cleared"))
    backup = pt_path.with_suffix(pt_path.suffix + ".prehead_reset")
    if not backup.exists():
        pt_path.rename(backup)
        torch.save(ckpt, pt_path)
        print(f"  {pt_path.name}: reset {len(changed)} items, backup {backup.name}")
    else:
        torch.save(ckpt, pt_path)
        print(f"  {pt_path.name}: reset {len(changed)} items (backup already existed)")
    for k, kind in changed[:8]:
        print(f"    {k}: {kind}")
    if len(changed) > 8:
        print(f"    (+{len(changed) - 8} more)")
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
