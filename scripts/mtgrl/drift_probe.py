"""Per-module relative weight drift between two checkpoints.

Usage: py -3.12 scripts/mtgrl/drift_probe.py [baseline_ckpt] [current_ckpt]
Defaults: May-31 Spy baseline vs the live Spy model_latest.pt.
Groups: critic_* / value_scale = critic head; transformer_layers/input_proj/
cross_attn/cand_* = shared encoder; policy_scorer* = policy heads.
"""
import sys
import collections
import torch

BASE = sys.argv[1] if len(sys.argv) > 1 else \
    'local-training/backups/spy_value_baseline_20260531/model_latest.pt'
CUR = sys.argv[2] if len(sys.argv) > 2 else \
    'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Spy-Combo-Value/models/model_latest.pt'

a = torch.load(BASE, map_location='cpu', weights_only=False)
b = torch.load(CUR, map_location='cpu', weights_only=False)
sa, sb = a['state_dict'], b['state_dict']

groups = collections.defaultdict(lambda: [0.0, 0.0])
for k in sa:
    if k not in sb or sa[k].shape != sb[k].shape:
        continue
    g = k.split('.')[0]
    groups[g][0] += (sb[k].float() - sa[k].float()).norm().item() ** 2
    groups[g][1] += sa[k].float().norm().item() ** 2

CRITIC = ('critic_proj1', 'critic_proj2', 'critic_norm', 'critic_norm1', 'value_scale')
ENCODER = ('transformer_layers', 'input_proj', 'input_norm', 'cross_attn',
           'cand_self_attn', 'cand_self_attn_norm', 'cand_feat_proj', 'cls_token')


def agg(names):
    d = sum(groups[n][0] for n in names if n in groups)
    n = sum(groups[n][1] for n in names if n in groups)
    return (d ** 0.5) / max(1e-9, n ** 0.5)


policy = [g for g in groups if g.startswith('policy_scorer')]
print(f"steps: {a.get('train_step_counter')} -> {b.get('train_step_counter')}")
print(f"critic  rel_drift = {agg(CRITIC):.4f}")
print(f"encoder rel_drift = {agg(ENCODER):.4f}")
print(f"policy  rel_drift = {agg(policy):.4f}")
rows = sorted(groups.items(), key=lambda kv: -(kv[1][0] ** 0.5 / max(1e-9, kv[1][1] ** 0.5)))
for g, (d, n) in rows[:6]:
    print(f"  top: {g:38s} {d ** 0.5 / max(1e-9, n ** 0.5):.4f}")
