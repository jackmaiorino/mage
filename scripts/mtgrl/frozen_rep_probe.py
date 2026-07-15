"""Frozen-representation probe (Codex second-opinion idea, 2026-06-17).

Question: does the TRAINED 128 encoder ALREADY contain the good-vs-dominated
Spy-cast distinction, such that a small linear probe on its frozen
representation can rank the search-best candidate above the search-worst?

If YES (probe acc >> 50%): the info is in the representation -> the candidate_q
failure is a HEAD/OBJECTIVE problem (shallow detached reranker), NOT capacity ->
a bigger net is the wrong fix; fix the head.
If NO (probe acc ~50%): the info is not linearly decodable from the frozen rep
-> encoder/capacity/objective genuinely implicated -> bigger-net / aux justified.

Probes the `combined`=[CLS, attended_candidate] vector that actually feeds the
policy/Q scorer, captured via a forward hook on the scorer module.

Usage: py -3.12 scripts/mtgrl/frozen_rep_probe.py [model.pt] [dump_dir]
"""
import sys
import glob

import numpy as np
import torch

sys.path.insert(0, 'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode')
from mtg_transformer import MTGTransformerModel  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'local-training/backups/candq_arm_20260612/model_best.pt'
DUMP = sys.argv[2] if len(sys.argv) > 2 else 'local-training/candq_dumps_v6'

ck = torch.load(MODEL, map_location='cpu', weights_only=False)
sd = ck['state_dict']
d_model = int(sd['cls_token'].shape[-1]) if 'cls_token' in sd else 128
nl = len({k.split('.')[1] for k in sd if k.startswith('transformer_layers.')}) or 2
dff = int(sd['transformer_layers.0.linear1.weight'].shape[0]) if 'transformer_layers.0.linear1.weight' in sd else 512
nhead = 8 if d_model % 8 == 0 else 4
print(f'model d_model={d_model} layers={nl} dff={dff} nhead={nhead}  ({MODEL})')
m = MTGTransformerModel(d_model=d_model, nhead=nhead, num_layers=nl, dim_feedforward=dff, cand_feat_dim=48)
m.load_state_dict(sd, strict=False)
m.eval()

# Capture the scorer INPUT (`combined` = [CLS, attended_candidate]) via a hook.
captured = {}
def hook(mod, inp, out):
    captured['combined'] = inp[0].detach()
# the action policy scorer is model.policy_scorer
h = m.policy_scorer.register_forward_hook(hook)

files = sorted(glob.glob(f'{DUMP}/*.npz'))
X, y = [], []
rows = 0
with torch.inference_mode():
    for f in files:
        d = np.load(f)
        if int(d['head_ids'][0]) != 0:  # action head only
            continue
        B = d['mcts_visits'].shape[0]
        for b in range(B):
            r = d['mcts_visits'][b]
            valid = (r >= -1.0) & (r <= 1.0)
            if valid.sum() < 2:
                continue
            best_i = int(np.where(valid, r, -9).argmax())
            worst_i = int(np.where(valid, r, 9).argmin())
            if r[best_i] - r[worst_i] < 1e-6:
                continue  # no discriminating gap
            captured.clear()
            m.score_candidates(
                torch.tensor(d['seq'][b:b+1].astype(np.float32)),
                torch.tensor(d['mask'][b:b+1].astype(np.bool_)),
                torch.tensor(d['tok_ids'][b:b+1].astype(np.int64)),
                torch.tensor(d['cand_feat'][b:b+1].astype(np.float32)),
                torch.tensor(d['cand_ids'][b:b+1].astype(np.int64)),
                torch.tensor(d['cand_mask'][b:b+1].astype(np.bool_)),
                'action', 0, 0, 0)
            comb = captured['combined'][0]  # [N, 2*d_model]
            X.append(comb[best_i].numpy()); y.append(1)
            X.append(comb[worst_i].numpy()); y.append(0)
            rows += 1

X = np.array(X, dtype=np.float64); y = np.array(y)
print(f'discriminating rows: {rows}  probe samples: {len(y)}  feature dim: {X.shape[1] if len(X) else 0}')
if rows < 30:
    print('too few rows for a reliable probe'); sys.exit(0)

h.remove()
# Torch logistic-regression probe, 5-fold CV (paired best/worst kept in same fold).
torch.manual_seed(0)
n_pairs = len(y) // 2  # rows are [best,worst, best,worst, ...]
pair_idx = np.arange(n_pairs)
rng = np.random.default_rng(0); rng.shuffle(pair_idx)
folds = np.array_split(pair_idx, 5)
Xt = torch.tensor(X, dtype=torch.float32)
mu, sd_ = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp_min(1e-6)
Xt = (Xt - mu) / sd_
yt = torch.tensor(y, dtype=torch.float32)
accs, aucs = [], []
for k in range(5):
    te_pairs = folds[k]; tr_pairs = np.concatenate([folds[j] for j in range(5) if j != k])
    te = np.concatenate([[2*p, 2*p+1] for p in te_pairs]); tr = np.concatenate([[2*p, 2*p+1] for p in tr_pairs])
    Xtr, ytr, Xte, yte = Xt[tr], yt[tr], Xt[te], yt[te]
    w = torch.zeros(Xt.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        logit = Xtr @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, ytr) + 1e-3 * (w*w).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(Xte @ w + b).numpy()
    accs.append(float(((p > 0.5).astype(float) == yte.numpy()).mean()))
    pos, neg = p[yte.numpy() == 1], p[yte.numpy() == 0]
    aucs.append(float((pos[:, None] > neg[None, :]).mean()))  # rank-AUC
acc_m, auc_m = np.mean(accs), np.mean(aucs)
print(f'\n=== FROZEN-REP LINEAR PROBE (best vs worst Spy-cast candidate) ===')
print(f'5-fold accuracy: {acc_m:.3f} +/- {np.std(accs):.3f}   (50% = rep has NO info; >65% = rep HAS it)')
print(f'5-fold ROC-AUC:  {auc_m:.3f} +/- {np.std(aucs):.3f}')
print('LINEAR VERDICT:', 'REP HAS THE INFO (linear) -> head/objective bottleneck' if acc_m > 0.62
      else ('AMBIGUOUS' if acc_m > 0.55 else 'rep NOT linearly separable'))

# Nonlinear (2-layer MLP) probe -- does a better head extract it where linear can't?
maccs = []
for k in range(5):
    te_pairs = folds[k]; tr_pairs = np.concatenate([folds[j] for j in range(5) if j != k])
    te = np.concatenate([[2*p, 2*p+1] for p in te_pairs]); tr = np.concatenate([[2*p, 2*p+1] for p in tr_pairs])
    Xtr, ytr, Xte, yte = Xt[tr], yt[tr], Xt[te], yt[te]
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr).squeeze(-1), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(net(Xte).squeeze(-1)).numpy()
    maccs.append(float(((p > 0.5).astype(float) == yte.numpy()).mean()))
macc = np.mean(maccs)
print(f'5-fold MLP accuracy: {macc:.3f} +/- {np.std(maccs):.3f}')
print('MLP VERDICT:', 'REP HAS THE INFO nonlinearly -> a BETTER HEAD fixes it (no bigger net)' if macc > 0.62
      else ('AMBIGUOUS' if macc > 0.55 else 'REP GENUINELY LACKS THE INFO -> encoder/capacity/input is the bottleneck'))
