"""Diagnostic: load current Standard-Wide model and print value-head parameter stats.

Answers: is value_scale collapsed to its 0.01 floor? Are critic_proj1/2 weights
shrunk/dead? Is critic_norm running-stats broken?
"""
import os
import sys
import torch
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLCODE = os.path.join(REPO_ROOT, "Mage.Server.Plugins", "Mage.Player.AIRL", "src", "mage", "player", "ai", "rl", "MLPythonCode")
sys.path.insert(0, MLCODE)

PROFILE = "Pauper-Standard-Wide"
MODEL_PATH = os.path.join(REPO_ROOT, "Mage.Server.Plugins", "Mage.Player.AIRL", "src", "mage", "player", "ai", "rl", "profiles", PROFILE, "models", "model_latest.pt")

print(f"Loading {MODEL_PATH}")
ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
state = ckpt.get('state_dict', ckpt)

print("\n=== Value-head critical parameters ===")
if 'value_scale' in state:
    vs = float(state['value_scale'].item())
    print(f"value_scale          = {vs:.6f}  (clamp range [0.01, 10.0])")
    if vs < 0.1:
        print("  *** COLLAPSED near floor ***")
else:
    print("value_scale NOT FOUND — searching alternatives:")
    for k in state:
        if 'scale' in k.lower() and state[k].numel() == 1:
            print(f"  {k} = {float(state[k].item()):.6f}")

# critic_proj1 stats
for key in ['critic_proj1.weight', 'critic_proj2.weight', 'critic_norm.weight', 'critic_norm1.weight']:
    if key in state:
        w = state[key]
        print(f"\n{key}: shape={tuple(w.shape)}")
        print(f"  mean={w.mean().item():+.4f}  std={w.std().item():.4f}  abs_max={w.abs().max().item():.4f}")
        # Dead neurons = rows whose magnitudes are all near zero
        if w.dim() == 2:
            row_norms = w.norm(dim=1)
            dead = (row_norms < 0.01).sum().item()
            print(f"  row_norms: min={row_norms.min().item():.4f} max={row_norms.max().item():.4f}  dead_rows={dead}/{row_norms.numel()}")

# bias stats
for key in ['critic_proj1.bias', 'critic_proj2.bias', 'critic_norm.bias', 'critic_norm1.bias']:
    if key in state:
        w = state[key]
        print(f"\n{key}: mean={w.mean().item():+.4f} std={w.std().item():.4f} abs_max={w.abs().max().item():.4f}")

print("\n=== Attention scales (for completeness) ===")
for k in state:
    if k.endswith('.scale') and state[k].numel() == 1:
        print(f"  {k:40s} = {float(state[k].item()):+.6f}")

print("\n=== Other interesting single-scalars ===")
for k in state:
    if state[k].numel() == 1 and not k.endswith('.scale'):
        v = float(state[k].item())
        if abs(v) > 0.01 or 'temp' in k.lower() or 'scale' in k.lower():
            print(f"  {k:40s} = {v:+.6f}")
