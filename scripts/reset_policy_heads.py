"""Reset policy scorer heads to fresh random init.

Motivation: the encoder and value head are now working correctly (shared-
encoder refactor + scale collapse fix). But the policy scorer MLPs
(policy_scorer, policy_scorer_target, etc.) were trained for 37k+ episodes
against a BROKEN value head. The advantages they saw were meaningless, so
they learned naive heuristics like "opponent-target = destructive = good"
for Cleansing Wildfire, even though the correct play is to target own
artifact land. Retraining the policy on top of the (now working) critic is
faster than nudging the entrenched bias out of a bad policy.

Keeps intact:
- Encoder (input_proj, token_id_emb, transformer_layers, cls_token,
  input_scale, input_norm): useful learned features.
- Cross/cand attention (cross_attn*, cand_self_attn*): useful.
- Candidate feature projection (cand_feat_proj, action_id_emb): useful.
- Critic head (critic_norm, critic_norm1, critic_proj1, critic_proj2,
  value_scale): now working.
- Legacy critic encoder (critic_transformer etc., unused but in state_dict).
- Temperature parameter.

Resets:
- policy_scorer (action head MLP)
- policy_scorer_target, policy_scorer_card_select, policy_scorer_attack,
  policy_scorer_block
- actor_norm, actor_proj1, actor_norm1, actor_proj2 (legacy actor head,
  reset for cleanliness)
- Adam optimizer state (momentum from old gradients would fight re-init)

Run: py -3.12 scripts/reset_policy_heads.py
Optional: PATCH_PROFILES="Pauper-Wildfire,Pauper-Rally,Pauper-Affinity,Pauper-Elves"
"""
import os
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_ROOT = REPO_ROOT / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "profiles"

POLICY_PREFIXES = (
    "policy_scorer.",
    "policy_scorer_target.",
    "policy_scorer_card_select.",
    "policy_scorer_attack.",
    "policy_scorer_block.",
    "actor_norm.",
    "actor_norm1.",
    "actor_proj1.",
    "actor_proj2.",
)


def classify(key: str, tensor: torch.Tensor) -> str:
    # Shape-based: 1D means LayerNorm params (indexed sequentials hide names).
    if key.endswith(".weight"):
        return "layernorm_weight" if tensor.dim() == 1 else "linear_weight"
    if key.endswith(".bias"):
        return "layernorm_bias" if tensor.dim() == 1 else "linear_bias"
    return "unknown"


def fresh_init(tensor: torch.Tensor, kind: str) -> torch.Tensor:
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
    return tensor


def patch_checkpoint(pt_path: Path) -> bool:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        print(f"  SKIP {pt_path}: unexpected format")
        return False
    sd = ckpt["state_dict"]
    changed = []
    for k in list(sd.keys()):
        if not any(k.startswith(p) for p in POLICY_PREFIXES):
            continue
        v = sd[k]
        kind = classify(k, v)
        if kind == "unknown":
            continue
        sd[k] = fresh_init(v, kind)
        changed.append((k, kind))
    if not changed:
        print(f"  {pt_path.name}: no policy-head keys found")
        return False
    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        ckpt["optimizer_state_dict"] = None
        changed.append(("optimizer_state_dict", "cleared"))
    backup = pt_path.with_suffix(pt_path.suffix + ".prepolicy_reset")
    if not backup.exists():
        pt_path.rename(backup)
        torch.save(ckpt, pt_path)
        print(f"  {pt_path.name}: reset {len(changed)} items, backup {backup.name}")
    else:
        torch.save(ckpt, pt_path)
        print(f"  {pt_path.name}: reset {len(changed)} items (backup existed)")
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
