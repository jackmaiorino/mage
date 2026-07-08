#!/usr/bin/env python3
"""
Belief probe (Codex #69): does the public event history predict the opponent's
HIDDEN HAND beyond the current-state snapshot?

Temporal modeling is fundamentally a belief/hidden-info hypothesis: in imperfect
info, the event sequence tells you what the opponent likely holds. This probe tests
the prerequisite -- if history carries no belief signal beyond the snapshot, the
temporal lever is weak.

Target: multi-hot over the opponent card vocab = which cards are in opp's hand NOW
(opp_hand_cards, a diagnostic label never fed to the agent).
Arms: S (snapshot facts+tokens+matchup) / S+H (S + event history) / H (history only).
The S-vs-S+H DIFFERENCE isolates history's marginal belief contribution (matchup-
trivial signal is in both). Also reported within-matchup (pure belief, no matchup-ID).

Usage: py -3.12 scripts/belief_probe.py [--epochs 20] [--arms S,S+H,H]
"""
import argparse, glob, json, math, random, os, sys
from collections import Counter
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history_probe import (load_rows, slice_of, Featurizer, SnapEncoder, HistEncoder,
                           collate as x_collate, build_vocabs, auc)


def build_opp_vocab(rows, min_count=40):
    c = Counter()
    for r in rows:
        for name in set(r.get("opp_hand_cards", [])):  # presence, not multiplicity
            c[name] += 1
    vocab = [name for name, n in c.most_common() if n >= min_count]
    return vocab


class BeliefProbe(nn.Module):
    def __init__(self, scalar_dim, tok_mod, ev_vocab, n_cards, arm):
        super().__init__()
        self.arm = arm
        self.has_snap = (arm != "H")
        self.has_hist = (arm != "S")
        d = 0
        if self.has_snap:
            self.snap = SnapEncoder(scalar_dim, tok_mod); d += 96
        if self.has_hist:
            self.hist = HistEncoder(ev_vocab, gru=(arm == "S+H+GRU")); d += self.hist.out_d
        self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_cards))

    def forward(self, S, tok, tok_mask, et, ecard, enum, emask):
        parts = []
        if self.has_snap:
            parts.append(self.snap(S, tok, tok_mask))
        if self.has_hist:
            parts.append(self.hist(et, ecard, enum, emask))
        z = torch.cat(parts, -1) if len(parts) > 1 else parts[0]
        return self.head(z)  # (B, n_cards) logits


def macro_auc(Y, P, min_pos=10, min_neg=10):
    """mean per-card AUC over cards with enough +/- in this eval set."""
    aucs = []
    for c in range(Y.shape[1]):
        yc = Y[:, c]
        if yc.sum() >= min_pos and (len(yc) - yc.sum()) >= min_neg:
            aucs.append(auc(yc, P[:, c]))
    aucs = [a for a in aucs if not math.isnan(a)]
    return (float(np.mean(aucs)), len(aucs)) if aucs else (float("nan"), 0)


def micro_bce(Y, P):
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return float(-(Y * np.log(P) + (1 - Y) * np.log(1 - P)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="E:/mage-training/corpus/affinity/*.jsonl")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="S,S+H,H")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    rows = load_rows(args.glob)
    rows = [r for r in rows if "opp_hand_cards" in r]
    print(f"loaded {len(rows)} rows with opp_hand_cards")
    if len(rows) < 500:
        print("too few rows; wait for more corpus."); return

    vocab = build_opp_vocab(rows)
    vidx = {n: i for i, n in enumerate(vocab)}
    print(f"opp-card vocab: {len(vocab)} cards (>=40 occurrences)")
    print("matchups:", Counter(r.get("matchup") for r in rows).most_common())
    matchups, ev_vocab = build_vocabs(rows)
    feat = Featurizer(matchups, ev_vocab)

    def multihot(r):
        mh = np.zeros(len(vocab), dtype=np.float32)
        for name in r.get("opp_hand_cards", []):
            j = vidx.get(name)
            if j is not None:
                mh[j] = 1.0
        return mh

    def prep(r):
        tok, tokm = feat.toks(r); et, ec, en, em = feat.events(r)
        return {"s": feat.scalars(r), "tok": tok, "tokm": tokm, "et": et, "ec": ec,
                "en": en, "em": em, "y": 0.0, "mh": multihot(r),
                "slice": slice_of(r), "matchup": r.get("matchup", "?"),
                "opp_hand": r.get("opp_hand", 0)}

    games = sorted({r["game"] for r in rows})
    random.Random(args.seed).shuffle(games)
    cut = int(len(games) * 0.8)
    train_g = set(games[:cut]); val_g = set(games[cut:])
    train = [prep(r) for r in rows if r["game"] in train_g]
    val = [prep(r) for r in rows if r["game"] in val_g]
    print(f"train {len(train)} ({len(train_g)} games) | val {len(val)} ({len(val_g)} games)")

    MH_tr = np.stack([b["mh"] for b in train]); MH_va = np.stack([b["mh"] for b in val])
    print(f"avg opp-hand cards in vocab per row: {MH_tr.sum(1).mean():.2f}")

    def run(arm):
        torch.manual_seed(args.seed)
        model = BeliefProbe(feat.scalar_dim, feat.tok_mod, len(ev_vocab), len(vocab), arm)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        bs = 128
        import time as _t
        t0 = _t.time()
        for ep in range(args.epochs):
            model.train(); idx = list(range(len(train))); random.shuffle(idx)
            for i in range(0, len(idx), bs):
                bidx = idx[i:i + bs]; b = [train[k] for k in bidx]
                S, tk, tm, et, ec, en, em, _ = x_collate(b)
                yb = torch.from_numpy(np.stack([train[k]["mh"] for k in bidx]))
                opt.zero_grad(); out = model(S, tk, tm, et, ec, en, em)
                loss = lossf(out, yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            S, tk, tm, et, ec, en, em, _ = x_collate(val)
            P = torch.sigmoid(model(S, tk, tm, et, ec, en, em)).numpy()
        print(f"[trained {arm} in {_t.time()-t0:.0f}s]", flush=True)
        return P

    preds = {arm: run(arm) for arm in arms}
    slices = np.array([b["slice"] for b in val])
    mus = np.array([b["matchup"] for b in val])
    ophand = np.array([b["opp_hand"] for b in val])

    def report(name, mask):
        if mask.sum() < 30:
            print(f"  {name:<22} n={int(mask.sum())} (too few)"); return
        Yv = MH_va[mask]
        cells = ""
        for a in arms:
            ma, ncard = macro_auc(Yv, preds[a][mask])
            bce = micro_bce(Yv, preds[a][mask])
            cells += f"  {a}:mAUC={ma:.3f}/BCE={bce:.4f}"
        print(f"  {name:<22} n={int(mask.sum()):>4}{cells}")
        if "S" in arms:
            hist_arms = [a for a in arms if a != "S"]
            if hist_arms:
                mS, _ = macro_auc(Yv, preds["S"][mask]); cS = micro_bce(Yv, preds["S"][mask])
                mH = max(macro_auc(Yv, preds[a][mask])[0] for a in hist_arms)
                cH = min(micro_bce(Yv, preds[a][mask]) for a in hist_arms)
                dr = (cS - cH) / cS * 100 if cS > 0 else 0
                flag = "  <-- HISTORY HELPS" if (mH - mS >= 0.03 or dr >= 5) else ""
                print(f"       dmAUC(bestH-S)={mH-mS:+.3f}  dBCE_rel={dr:+.1f}%{flag}")

    print("\n=== BELIEF PROBE: predict opp hidden hand (macro per-card AUC / micro BCE) ===")
    print("pooled (S has matchup+counts; S-vs-S+H isolates history's marginal belief signal)")
    report("all", np.ones(len(val), bool))
    report("combat", slices == "combat")
    report("response", slices == "response")
    report("opp_has_hand>=3", ophand >= 3)
    report("reactive&opphand>=2", ((slices == "combat") | (slices == "response")) & (ophand >= 2))

    print("\n=== within-matchup (pure belief, no matchup-ID shortcut) ===")
    for m in sorted(set(mus)):
        report(f"matchup={m}", mus == m)

    print("\nPASS (Codex #69): S+H beats S by >=3-5pp mAUC OR >=5% rel BCE on contested/reactive,")
    print("and holds within-matchup. Then the necessary-vs-sufficient guard: oracle-belief decision-relevance.")


if __name__ == "__main__":
    main()
