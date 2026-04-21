"""
Reset collapsed self_attn.scale parameters back to 1.0 in saved model checkpoints.

Training drove every self_attn.scale to 0.0, making attention uniform and
turning the transformer encoder into a bag-of-words averager. This script
patches each profile's model_latest.pt so the parameter starts at a usable
value; combined with SCALED_MHA_MIN_SCALE=1.0 in forward (the floor is a
safety net), the optimizer can then learn a proper scale.

Also renormalizes value_scale to 1.0 so the critic head has a clean baseline.

Run with: py -3.12 scripts/patch_scale_collapse.py
Optionally: PATCH_PROFILES="Pauper-Wildfire,..." to limit.
"""
import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_ROOT = REPO_ROOT / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "profiles"

SCALE_KEY_SUFFIX = ".self_attn.scale"
VALUE_SCALE_KEY = "value_scale"
INIT_SCALE = 1.0
INIT_VALUE_SCALE = 1.0


def patch_checkpoint(pt_path: Path) -> bool:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        print(f"  SKIP {pt_path}: unexpected format")
        return False
    sd = ckpt["state_dict"]
    changed = []
    for k, v in sd.items():
        if k.endswith(SCALE_KEY_SUFFIX) and v.numel() == 1:
            old = float(v.item())
            if abs(old - INIT_SCALE) > 1e-6:
                sd[k] = torch.full_like(v, INIT_SCALE)
                changed.append((k, old, INIT_SCALE))
        elif k == VALUE_SCALE_KEY and v.numel() == 1:
            old = float(v.item())
            if abs(old - INIT_VALUE_SCALE) > 0.05:
                sd[k] = torch.full_like(v, INIT_VALUE_SCALE)
                changed.append((k, old, INIT_VALUE_SCALE))
    # Drop stale Adam optimizer state. It holds momentum/variance for the
    # *old* scale gradient (wants to drive scale back to 0); carrying it
    # forward undoes the reset within a few updates.
    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        ckpt["optimizer_state_dict"] = None
        changed.append(("optimizer_state_dict", "saved", "cleared"))
    if not changed:
        print(f"  {pt_path.name}: already normalized, skipping")
        return False
    backup = pt_path.with_suffix(pt_path.suffix + ".prescale_fix")
    if not backup.exists():
        pt_path.rename(backup)
        torch.save(ckpt, pt_path)
        print(f"  {pt_path.name}: patched {len(changed)} params, backup at {backup.name}")
    else:
        torch.save(ckpt, pt_path)
        print(f"  {pt_path.name}: patched {len(changed)} params (backup already existed, kept)")
    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)
    for k, old, new in changed[:6]:
        print(f"    {k}: {_fmt(old)} -> {_fmt(new)}")
    if len(changed) > 6:
        print(f"    (+{len(changed) - 6} more)")
    return True


def main() -> None:
    filter_csv = os.getenv("PATCH_PROFILES", "")
    only = set(x.strip() for x in filter_csv.split(",") if x.strip())
    candidates = sorted(PROFILES_ROOT.glob("*/models/model_latest.pt"))
    patched = 0
    for ckpt in candidates:
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
