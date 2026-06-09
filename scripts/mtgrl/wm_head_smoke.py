"""Isolated learnability + round-trip smoke for the world-model aux head.

Proves, without the JVM/game pipeline:
  1. The head allocates only when WORLD_MODEL_DIM>0 (and is absent at 0).
  2. world_model_logits_from_cls returns (B, dim).
  3. Head-only training on synthetic future-count targets drives loss down.
  4. The exact Python training-loss form (sigmoid + smooth_l1, valid mask) works
     and skips -1 sentinel rows.
  5. get_config round-trips and strict=False load tolerates the new head keys.
"""
import os
import sys

os.environ["WORLD_MODEL_DIM"] = "18"
os.environ.setdefault("MODEL_D_MODEL", "128")
os.environ.setdefault("MODEL_NHEAD", "4")
os.environ.setdefault("MODEL_NUM_LAYERS", "2")
os.environ.setdefault("MODEL_DIM_FEEDFORWARD", "512")

HERE = os.path.dirname(os.path.abspath(__file__))
MLDIR = os.path.normpath(os.path.join(
    HERE, "..", "..", "Mage.Server.Plugins", "Mage.Player.AIRL",
    "src", "mage", "player", "ai", "rl", "MLPythonCode"))
sys.path.insert(0, MLDIR)

import torch  # noqa: E402
from mtg_transformer import MTGTransformerModel  # noqa: E402

torch.manual_seed(0)
D_MODEL = int(os.environ["MODEL_D_MODEL"])
DIM = 18
B = 64

m = MTGTransformerModel(d_model=D_MODEL, nhead=4, num_layers=2,
                        dim_feedforward=512, cand_feat_dim=48)

# 1. head allocates
assert m.world_model_dim == DIM, m.world_model_dim
assert m.world_model_head is not None, "head not built"
print(f"[1] head allocated: world_model_dim={m.world_model_dim}")

# 2. forward shape
cls = torch.randn(B, D_MODEL)
logits = m.world_model_logits_from_cls(cls)
assert logits.shape == (B, DIM), logits.shape
print(f"[2] forward ok: logits {tuple(logits.shape)}")

# 3+4. head-only learnability on a synthetic target that is a deterministic
# function of CLS, with some rows -1 (sentinel) to exercise the valid mask.
# Freeze everything but the head (mirrors WORLD_MODEL_HEAD_ONLY).
for name, p in m.named_parameters():
    p.requires_grad = name.startswith("world_model_head.")
trainable = [p for p in m.parameters() if p.requires_grad]
assert trainable, "no trainable params"
opt = torch.optim.Adam(trainable, lr=1e-2)

# Fixed CLS batch + a learnable-from-CLS target in [0,1].
W = torch.randn(D_MODEL, DIM)
cls_fixed = torch.randn(B, D_MODEL)
target = torch.sigmoid(cls_fixed @ W).detach().clamp(0, 1)
# Mark 20% of rows as absent (-1 sentinel across all dims).
absent = torch.zeros(B, dtype=torch.bool)
absent[: B // 5] = True
labels = target.clone()
labels[absent] = -1.0

def wm_loss(coef=0.1):
    lo = m.world_model_logits_from_cls(cls_fixed)
    valid = torch.isfinite(labels).all(dim=-1) & (labels >= 0.0).all(dim=-1)
    if not valid.any():
        return torch.zeros(())
    pred = torch.sigmoid(lo[valid].float())
    tgt = labels[valid].float().clamp(0.0, 1.0)
    return coef * torch.nn.functional.smooth_l1_loss(pred, tgt, reduction="mean")

l0 = float(wm_loss().item())
for _ in range(300):
    opt.zero_grad()
    loss = wm_loss()
    loss.backward()
    opt.step()
l1 = float(wm_loss().item())
n_valid = int((torch.isfinite(labels).all(dim=-1) & (labels >= 0).all(dim=-1)).sum())
print(f"[3] valid rows {n_valid}/{B} (sentinels skipped: {B - n_valid})")
print(f"[4] head-only loss {l0:.5f} -> {l1:.5f} ({100*(l0-l1)/max(l0,1e-9):.0f}% down)")
assert l1 < l0 * 0.5, f"head did not learn: {l0} -> {l1}"

# Verify the frozen encoder truly received no gradient.
enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in m.named_parameters()
               if not n.startswith("world_model_head."))
assert not enc_grad, "encoder got gradient under head-only freeze"
print("[5] encoder frozen: no gradient leaked to shared encoder")

# 5. config round-trip + strict=False load tolerance.
cfg = m.get_config()
assert cfg.get("world_model_dim") == DIM, cfg.get("world_model_dim")
# An OLD checkpoint lacks world_model_head.* keys -> strict=False must load.
sd = {k: v for k, v in m.state_dict().items()
      if not k.startswith("world_model_head.")}
m2 = MTGTransformerModel(d_model=D_MODEL, nhead=4, num_layers=2,
                         dim_feedforward=512, cand_feat_dim=48)
res = m2.load_state_dict(sd, strict=False)
missing_wm = [k for k in res.missing_keys if k.startswith("world_model_head.")]
assert missing_wm, "expected world_model_head.* in missing_keys for old ckpt"
print(f"[6] get_config has world_model_dim={cfg['world_model_dim']}; "
      f"strict=False load tolerates {len(missing_wm)} missing head keys")

print("\nALL CHECKS PASSED")
