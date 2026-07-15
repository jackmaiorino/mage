"""Analyze candidate_q search-target dumps: discrimination + does the trained
Q-head predict the search-best candidate?

Usage: py -3.12 scripts/mtgrl/candq_dump_analyze.py [dump_dir] [model.pt]
"""
import sys
import glob

import numpy as np
import torch

sys.path.insert(0, 'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode')
from mtg_transformer import MTGTransformerModel  # noqa: E402

import os
DUMP = sys.argv[1] if len(sys.argv) > 1 else 'local-training/candq_dumps_v6'
MODEL = sys.argv[2] if len(sys.argv) > 2 else 'local-training/backups/candq_arm_20260612/model_best.pt'
D_MODEL = int(os.getenv('MODEL_D_MODEL', '128'))
NUM_LAYERS = int(os.getenv('MODEL_NUM_LAYERS', '2'))
NHEAD = int(os.getenv('MODEL_NHEAD', '4'))

files = sorted(glob.glob(f'{DUMP}/*.npz'))
print(f'dump files: {len(files)}')

# ---- Part 1: target discrimination (signed targets in [-1,1], -2 = sentinel) ----
rows = 0
disc = 0          # rows with a real best/worst gap
gaps = []
target_vals = []
obs_counts = []
all_rows = []     # (seq,mask,tok,cf,cid,cm,targets) for rows with >=2 observed
for f in files:
    d = np.load(f)
    mv = d['mcts_visits']
    B = mv.shape[0]
    for b in range(B):
        r = mv[b]
        obs = r[(r >= -1.0) & (r <= 1.0)]   # observed targets (sentinel -2 excluded)
        if len(obs) < 2:
            continue
        rows += 1
        obs_counts.append(len(obs))
        g = float(obs.max() - obs.min())
        gaps.append(g)
        target_vals.extend(obs.tolist())
        if g > 1e-6:
            disc += 1
            all_rows.append((d, b))

gaps = np.array(gaps)
tv = np.array(target_vals)
print(f'\n=== TARGET DISCRIMINATION ===')
print(f'rows (>=2 observed candidates): {rows}')
print(f'discriminating (best>worst):    {disc} ({100*disc/max(1,rows):.1f}%)')
print(f'gap distribution: ', {float(k): int(v) for k, v in zip(*np.unique(np.round(gaps, 1), return_counts=True))})
print(f'target value spread: min={tv.min():.2f} max={tv.max():.2f} mean={tv.mean():.3f}')
print(f'  target sign: pos={int((tv>0.01).sum())} zero={int((np.abs(tv)<=0.01).sum())} neg={int((tv<-0.01).sum())}')
print(f'observed-per-row: ', {int(k): int(v) for k, v in zip(*np.unique(obs_counts, return_counts=True))})

# ---- Part 2: does the trained Q-head agree with the search-best candidate? ----
ckpt = torch.load(MODEL, map_location='cpu', weights_only=False)
# auto-detect d_model from the checkpoint (cls_token is [1,1,d_model]) so the
# probe works on both the 128 small net and the 256 wide net.
_sd = ckpt['state_dict']
if 'cls_token' in _sd:
    D_MODEL = int(_sd['cls_token'].shape[-1])
NUM_LAYERS = len({k.split('.')[1] for k in _sd if k.startswith('transformer_layers.')}) or NUM_LAYERS
DIM_FF = int(_sd['transformer_layers.0.linear1.weight'].shape[0]) if 'transformer_layers.0.linear1.weight' in _sd else 512
print(f'detected d_model={D_MODEL} num_layers={NUM_LAYERS} dim_ff={DIM_FF} nhead={NHEAD}')
m = MTGTransformerModel(d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FF, cand_feat_dim=48)
m.load_state_dict(ckpt['state_dict'], strict=False)
m.eval()
print(f'\n=== Q-HEAD vs SEARCH TARGETS (model: {MODEL}) ===')
print(f'train_step_counter: {ckpt.get("train_step_counter")}')

agree = 0
checked = 0
q_at_best = []
q_at_worst = []
pol_best = []
pol_worst = []
import random
random.seed(0)
sample = all_rows if len(all_rows) <= 600 else random.sample(all_rows, 600)
with torch.inference_mode():
    for d, b in sample:
        r = d['mcts_visits'][b]
        valid = (r >= -1.0) & (r <= 1.0)
        if valid.sum() < 2:
            continue
        seq = torch.tensor(d['seq'][b:b+1].astype(np.float32))
        mask = torch.tensor(d['mask'][b:b+1].astype(np.bool_))
        tok = torch.tensor(d['tok_ids'][b:b+1].astype(np.int64))
        cf = torch.tensor(d['cand_feat'][b:b+1].astype(np.float32))
        cid = torch.tensor(d['cand_ids'][b:b+1].astype(np.int64))
        cm = torch.tensor(d['cand_mask'][b:b+1].astype(np.bool_))
        out = m.score_candidates(seq, mask, tok, cf, cid, cm, 'action', 0, 0, 0,
                                 return_candidate_q=True, return_logits=True)
        probs = out[0][0].numpy()      # policy probs [64]
        q = out[-1][0].numpy()         # candidate_q [64]
        best_i = int(np.where(valid, r, -9).argmax())
        worst_i = int(np.where(valid, r, 9).argmin())
        checked += 1
        if int(np.where(valid, q, -9).argmax()) == best_i:
            agree += 1
        q_at_best.append(float(q[best_i]))
        q_at_worst.append(float(q[worst_i]))
        pol_best.append(float(probs[best_i]))
        pol_worst.append(float(probs[worst_i]))

qb, qw = np.array(q_at_best), np.array(q_at_worst)
pb, pw = np.array(pol_best), np.array(pol_worst)
print(f'\n=== POLICY representation check (can the encoder even separate these?) ===')
print(f'mean policy prob at search-best:  {pb.mean():.3f}')
print(f'mean policy prob at search-worst: {pw.mean():.3f}')
print(f'rows where policy(best) > policy(worst): {100*(pb>pw).mean():.1f}%  '
      f'[if ~50%, the representation does NOT separate them -> a detached Q-head CANNOT either]')
print(f'checked rows: {checked}')
print(f'Q-head argmax == search-best:   {agree} ({100*agree/max(1,checked):.1f}%)  [random ~ 1/obs]')
print(f'mean Q at search-best:  {qb.mean():.3f}')
print(f'mean Q at search-worst: {qw.mean():.3f}')
print(f'Q separation (best-worst): {qb.mean()-qw.mean():.3f}  [>0 means Q learned the ranking]')
# paired: fraction of rows where Q ranks best above worst
print(f'rows where Q(best) > Q(worst): {100*(qb>qw).mean():.1f}%  [50% = no signal, 100% = perfect]')
