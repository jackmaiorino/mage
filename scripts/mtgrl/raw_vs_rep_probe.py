"""Confound checks for the frozen-rep probe (Codex #2): label-noise filter +
raw-input oracle. Compares whether the good-vs-dominated Spy-cast label is
decodable from (a) the frozen encoder representation `combined`, vs (b) the RAW
pooled input the encoder sees -- on ALL pairs and on HIGH-CONFIDENCE pairs only.

Raw-input oracle succeeds  -> input sufficient + labels clean -> encoder is the bottleneck.
Raw-input oracle fails too  -> input insufficient OR labels noisy -> 'encoder bottleneck' undermined.
High-conf >> all-pairs       -> label noise was masking signal.

Usage: py -3.12 scripts/mtgrl/raw_vs_rep_probe.py [model.pt] [dump_dir]
"""
import sys, glob
import numpy as np
import torch
sys.path.insert(0, 'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode')
from mtg_transformer import MTGTransformerModel  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'local-training/backups/candq_arm_20260612/model_best.pt'
DUMP = sys.argv[2] if len(sys.argv) > 2 else 'local-training/candq_dumps_v6'
ck = torch.load(MODEL, map_location='cpu', weights_only=False); sd = ck['state_dict']
dm = int(sd['cls_token'].shape[-1]); nl = len({k.split('.')[1] for k in sd if k.startswith('transformer_layers.')}) or 2
dff = int(sd['transformer_layers.0.linear1.weight'].shape[0]); nh = 8 if dm % 8 == 0 else 4
m = MTGTransformerModel(d_model=dm, nhead=nh, num_layers=nl, dim_feedforward=dff, cand_feat_dim=48); m.load_state_dict(sd, strict=False); m.eval()
cap = {}; h = m.policy_scorer.register_forward_hook(lambda mod, i, o: cap.__setitem__('c', i[0].detach()))

REP_b, REP_w, RAW_b, RAW_w, GAP = [], [], [], [], []
with torch.inference_mode():
    for f in sorted(glob.glob(f'{DUMP}/*.npz')):
        d = np.load(f)
        if int(d['head_ids'][0]) != 0: continue
        for b in range(d['mcts_visits'].shape[0]):
            r = d['mcts_visits'][b]; valid = (r >= -1.0) & (r <= 1.0)
            if valid.sum() < 2: continue
            bi = int(np.where(valid, r, -9).argmax()); wi = int(np.where(valid, r, 9).argmin())
            gap = float(r[bi] - r[wi])
            if gap < 1e-6: continue
            seq = d['seq'][b:b+1].astype(np.float32); cf = d['cand_feat'][b:b+1].astype(np.float32)
            cap.clear()
            m.score_candidates(torch.tensor(seq), torch.tensor(d['mask'][b:b+1].astype(np.bool_)),
                torch.tensor(d['tok_ids'][b:b+1].astype(np.int64)), torch.tensor(cf),
                torch.tensor(d['cand_ids'][b:b+1].astype(np.int64)), torch.tensor(d['cand_mask'][b:b+1].astype(np.bool_)), 'action', 0, 0, 0)
            comb = cap['c'][0].numpy()
            sq = seq[0]; pooled = np.concatenate([sq.mean(0), sq.max(0)])  # raw input the encoder sees, pooled
            REP_b.append(comb[bi]); REP_w.append(comb[wi])
            RAW_b.append(np.concatenate([pooled, cf[0][bi]])); RAW_w.append(np.concatenate([pooled, cf[0][wi]]))
            GAP.append(gap)
REP_b, REP_w = np.array(REP_b), np.array(REP_w); RAW_b, RAW_w = np.array(RAW_b), np.array(RAW_w); GAP = np.array(GAP)
print(f'pairs: {len(GAP)}  high-conf (gap>=2, clean win vs loss): {(GAP>=2.0).sum()}')

def probe(Xb, Xw, name, mask=None):
    if mask is not None: Xb, Xw = Xb[mask], Xw[mask]
    n = len(Xb)
    if n < 30: print(f'  {name}: too few ({n})'); return
    X = np.concatenate([Xb, Xw]); y = np.concatenate([np.ones(n), np.zeros(n)])
    Xt = torch.tensor(X, dtype=torch.float32); mu, s = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp_min(1e-6); Xt = (Xt-mu)/s
    yt = torch.tensor(y, dtype=torch.float32)
    idx = np.arange(n); rng = np.random.default_rng(0); rng.shuffle(idx); folds = np.array_split(idx, 5)
    accs = []
    for k in range(5):
        tep = folds[k]; trp = np.concatenate([folds[j] for j in range(5) if j != k])
        te = np.concatenate([tep, tep+n]); tr = np.concatenate([trp, trp+n])
        net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(400):
            opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[tr]).squeeze(-1), yt[tr]); loss.backward(); opt.step()
        with torch.no_grad(): p = torch.sigmoid(net(Xt[te]).squeeze(-1)).numpy()
        accs.append(float(((p > 0.5).astype(float) == yt[te].numpy()).mean()))
    print(f'  {name}: MLP acc {np.mean(accs):.3f} +/- {np.std(accs):.3f}  (n={n})')

hc = GAP >= 2.0
print('FROZEN-REP probe:'); probe(REP_b, REP_w, 'all pairs'); probe(REP_b, REP_w, 'high-conf only', hc)
print('RAW-INPUT oracle (pooled seq + cand_feat):'); probe(RAW_b, RAW_w, 'all pairs'); probe(RAW_b, RAW_w, 'high-conf only', hc)
h.remove()
