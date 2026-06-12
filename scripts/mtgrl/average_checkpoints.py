"""Average the float tensors of several training checkpoints (Polyak-style).

Usage: py -3.12 scripts/mtgrl/average_checkpoints.py out.pt ck1.pt ck2.pt ...
Non-float tensors and metadata (counters, optimizer state) are taken from the
LAST checkpoint given, so continue-training semantics stay coherent.
"""
import sys
import torch

out_path, paths = sys.argv[1], sys.argv[2:]
assert len(paths) >= 2, "need at least 2 checkpoints"

base = torch.load(paths[-1], map_location='cpu', weights_only=False)
acc = {k: v.float().clone() for k, v in base['state_dict'].items()
       if torch.is_floating_point(v)}
n = 1
for p in paths[:-1]:
    sd = torch.load(p, map_location='cpu', weights_only=False)['state_dict']
    ok = True
    for k in acc:
        if k not in sd or sd[k].shape != acc[k].shape:
            ok = False
            break
    if not ok:
        print(f'SKIP (shape mismatch): {p}')
        continue
    for k in acc:
        acc[k] += sd[k].float()
    n += 1
for k in acc:
    base['state_dict'][k] = (acc[k] / n).to(base['state_dict'][k].dtype)
torch.save(base, out_path)
print(f'averaged {n} checkpoints -> {out_path}')
