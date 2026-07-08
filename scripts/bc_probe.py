#!/usr/bin/env python3
"""
Phase-0 offline BC sanity probe (Codex #74d): are shadow-CP7 teacher labels
LEARNABLE from our encoding, and does a learned ranker beat the CURRENT AGENT
POLICY at predicting CP7's choice on held-out games?

Data: shadow_v2 corpus rows (ACTIVATE, cp7_idx >= 0, timeout_pass excluded,
mana-tap-chosen rows excluded). Each row has the full candidate set: per-candidate
encoder features (cand_feats), the agent's policy distribution (probs), and the
CP7 teacher label (cp7_idx).

Model: state embedding (scalars + card tokens, reuses history_probe.SnapEncoder)
+ per-candidate MLP on cand_feats, scored jointly -> softmax over candidates ->
NLL of cp7_idx. Baselines: agent policy probs (the reference to beat), uniform.

PASS (pre-registered): held-out CP7-label CE >= 5% rel better than the agent-policy
baseline on target rows AND top-1 agreement with CP7 materially above the agent's
own agreement. Report per matchup + pass/non-pass slices.
"""
import argparse, glob, json, math, os, random, sys
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history_probe import Featurizer, SnapEncoder, build_vocabs, load_rows


def usable(r):
    if r.get('dtype') != 'ACTIVATE_ABILITY_OR_SPELL':
        return False
    if r.get('cp7_idx', -2) < 0:
        return False  # not queried / unmatched / timeout_pass
    if '{T}: Add' in (r.get('chosen_desc') or ''):
        return False  # mana-payment context: CP7 plans at a different abstraction
    cf = r.get('cand_feats')
    if not cf or len(cf) < r.get('n_cand', 0):
        return False
    if r.get('cp7_idx', -1) >= r.get('n_cand', 0):
        return False
    return True


class BCRanker(nn.Module):
    def __init__(self, scalar_dim, tok_mod, cand_dim):
        super().__init__()
        self.snap = SnapEncoder(scalar_dim, tok_mod)  # -> 96
        self.cand = nn.Sequential(nn.Linear(cand_dim, 96), nn.ReLU(), nn.Linear(96, 96))
        self.score = nn.Sequential(nn.Linear(96 * 2, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, S, tok, tokm, CF, cmask):
        # S: (B, sd), CF: (B, K, cd), cmask: (B, K) 1=real
        z = self.snap(S, tok, tokm)                       # (B, 96)
        c = self.cand(CF)                                 # (B, K, 96)
        zz = z.unsqueeze(1).expand(-1, CF.shape[1], -1)   # (B, K, 96)
        logits = self.score(torch.cat([zz, c], -1)).squeeze(-1)  # (B, K)
        return logits.masked_fill(cmask == 0, -1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='E:/mage-training/corpus/shadow_v2/*.jsonl')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--kmax', type=int, default=16)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    rows = [r for r in load_rows(args.glob) if usable(r)]
    print(f'usable BC rows: {len(rows)}')
    print('by matchup:', Counter(r.get('matchup') for r in rows).most_common())
    print('label=pass share:', np.mean([1.0 if r['cp7_idx'] == 0 else 0.0 for r in rows]).round(3))
    if len(rows) < 2000:
        print('too few rows; wait for more corpus.'); return

    cand_dim = len(rows[0]['cand_feats'][0])
    print(f'cand_feats dim: {cand_dim}, kmax: {args.kmax}')
    matchups, ev_vocab = build_vocabs(rows)
    feat = Featurizer(matchups, ev_vocab)

    K = args.kmax

    def prep(r):
        tok, tokm = feat.toks(r)
        n = min(r['n_cand'], K)
        CF = np.zeros((K, cand_dim), dtype=np.float32)
        for i in range(n):
            f = r['cand_feats'][i]
            CF[i, :min(len(f), cand_dim)] = f[:cand_dim]
        cmask = np.zeros(K, dtype=np.float32); cmask[:n] = 1
        # agent policy baseline (renormalized over first K)
        pr = np.zeros(K, dtype=np.float32)
        probs = r.get('probs') or []
        for i in range(min(len(probs), n)):
            pr[i] = probs[i] if probs[i] is not None else 0.0
        s = pr.sum()
        pr = pr / s if s > 1e-9 else cmask / max(1, cmask.sum())
        y = min(r['cp7_idx'], K - 1)
        agent_choice = min(r['chosen'], K - 1)
        return {'s': feat.scalars(r), 'tok': tok, 'tokm': tokm, 'CF': CF, 'cmask': cmask,
                'y': y, 'agent_probs': pr, 'agent_choice': agent_choice,
                'matchup': r.get('matchup', '?'), 'is_pass': r['cp7_idx'] == 0,
                'game': r['game']}

    games = sorted({r['game'] for r in rows})
    random.Random(args.seed).shuffle(games)
    cut = int(len(games) * 0.8)
    train_g, val_g = set(games[:cut]), set(games[cut:])
    train = [prep(r) for r in rows if r['game'] in train_g]
    val = [prep(r) for r in rows if r['game'] in val_g]
    print(f'train {len(train)} ({len(train_g)} games) | val {len(val)} ({len(val_g)} games)')

    def collate(b):
        S = torch.from_numpy(np.stack([x['s'] for x in b]))
        tok = torch.from_numpy(np.stack([x['tok'] for x in b]))
        tokm = torch.from_numpy(np.stack([x['tokm'] for x in b]))
        CF = torch.from_numpy(np.stack([x['CF'] for x in b]))
        cmask = torch.from_numpy(np.stack([x['cmask'] for x in b]))
        y = torch.tensor([x['y'] for x in b], dtype=torch.long)
        return S, tok, tokm, CF, cmask, y

    model = BCRanker(feat.scalar_dim, feat.tok_mod, cand_dim)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    bs = 128
    import time as _t
    for ep in range(args.epochs):
        t0 = _t.time(); model.train(); random.shuffle(train); tot = 0.0; nb = 0
        for i in range(0, len(train), bs):
            b = train[i:i + bs]
            S, tok, tokm, CF, cmask, y = collate(b)
            opt.zero_grad()
            loss = lossf(model(S, tok, tokm, CF, cmask), y)
            loss.backward(); opt.step(); tot += float(loss); nb += 1
        print(f'[ep {ep+1}/{args.epochs} loss {tot/max(1,nb):.4f} {_t.time()-t0:.0f}s]', flush=True)

    model.eval()
    with torch.no_grad():
        S, tok, tokm, CF, cmask, y = collate(val)
        logits = model(S, tok, tokm, CF, cmask)
        logp = torch.log_softmax(logits, -1)
        pred = logits.argmax(-1).numpy()
    yv = y.numpy()
    nllm = -logp[torch.arange(len(yv)), y].numpy()
    agent_probs = np.stack([x['agent_probs'] for x in val])
    agent_nll = -np.log(np.clip(agent_probs[np.arange(len(yv)), yv], 1e-6, 1))
    agent_top1 = np.array([x['agent_choice'] for x in val]) == yv
    mus = np.array([x['matchup'] for x in val])
    isp = np.array([x['is_pass'] for x in val])
    unif_nll = -np.log(1.0 / np.maximum(1, np.stack([x['cmask'] for x in val]).sum(1)))

    def report(name, mask):
        if mask.sum() < 50:
            print(f'  {name:<22} n={int(mask.sum())} (too few)'); return
        ce_m, ce_a, ce_u = nllm[mask].mean(), agent_nll[mask].mean(), unif_nll[mask].mean()
        t1_m = (pred[mask] == yv[mask]).mean()
        t1_a = agent_top1[mask].mean()
        rel = (ce_a - ce_m) / ce_a * 100 if ce_a > 0 else 0
        flag = '  <-- LEARNABLE' if (rel >= 5 and t1_m > t1_a) else ''
        print(f'  {name:<22} n={int(mask.sum()):>5} CE ranker={ce_m:.4f} agent={ce_a:.4f} unif={ce_u:.4f} '
              f'rel={rel:+.1f}% | top1 ranker={t1_m:.3f} agent={t1_a:.3f}{flag}')

    print('\n=== PHASE-0 BC PROBE: predict CP7 choice on held-out games ===')
    report('all', np.ones(len(yv), bool))
    report('non-pass', ~isp)
    report('pass', isp)
    print('\n=== by matchup ===')
    for m in sorted(set(mus)):
        report(f'{m}', mus == m)
        report(f'{m} non-pass', (mus == m) & ~isp)
    print('\nPASS bar: ranker CE >=5% rel better than agent-policy CE AND top1 > agent top1,')
    print('on target rows, holding per-matchup. -> proceed to Phase-1 BC warm-start in the training stack.')


if __name__ == '__main__':
    main()
