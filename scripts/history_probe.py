#!/usr/bin/env python3
"""
Temporal-history falsifier (Step 1) probe.

Tests whether the last-K public event history carries outcome-predictive signal
BEYOND the current-state snapshot, on the competent Affinity agent's decisions.

Three arms (frozen-feature probes; the "backbone" here is the hand-built snapshot):
  S        : snapshot state-facts + card-identity token bag
  S+H      : S + mean-pooled event-history embedding
  S+H+GRU  : S + GRU-encoded event-history (expressive head)

Target: terminal outcome (won) -- thesis-clean (Codex: weak but no labels).
Controls (Codex #67): S already contains turn + matchup one-hot, so any S+H gain
is provably NOT a turn-number / matchup-ID proxy. Reported PER SLICE (combat /
response / easy-main) with game-level held-out split (no row leakage).

Usage: py -3.12 scripts/history_probe.py [--glob 'E:/mage-training/corpus/affinity/*.jsonl'] [--epochs 30]
"""
import argparse, glob, json, math, random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

# ---------------- data ----------------
PHASES = ["BEGINNING", "PRECOMBAT_MAIN", "COMBAT", "POSTCOMBAT_MAIN", "END", ""]
EVENT_TYPES = ["cast", "spellcast", "land", "attack", "block", "dmg", "counter", "zone", "<pad>", "<unk>"]
TOK_MAX = 64
EV_MAX = 32
# NOTE: 'value' (the model's own value-head estimate) is deliberately EXCLUDED --
# it is a trained outcome predictor, so including it leaks the label into the S arm
# and masks any marginal history signal. Snapshot = raw public state facts only.
SCALAR_KEYS = ["my_life", "opp_life", "my_hand", "opp_hand", "my_lib", "my_gy", "opp_gy", "stack",
               "my_cre", "my_pow", "my_land", "my_untap_land", "my_art", "my_tap_cre",
               "opp_cre", "opp_pow", "opp_land", "opp_untap_land", "opp_art", "opp_tap_cre",
               "turn", "n_cand"]


def load_rows(pattern):
    rows = []
    for f in glob.glob(pattern):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def build_vocabs(rows):
    matchups = sorted({r.get("matchup", "unknown") for r in rows})
    ev_cards = Counter()
    for r in rows:
        for e in r.get("events", []):
            ev_cards[e.get("c", "")] += 1
    # keep frequent event card names
    ev_vocab = {"<pad>": 0, "<unk>": 1}
    for c, _ in ev_cards.most_common(400):
        ev_vocab.setdefault(c, len(ev_vocab))
    return matchups, ev_vocab


def slice_of(r):
    dt = r.get("dtype", "")
    if dt in ("DECLARE_ATTACKS", "DECLARE_BLOCKS"):
        return "combat"
    if r.get("stack", 0) > 0 or not r.get("my_turn", True):
        return "response"
    return "main_easy"


class Featurizer:
    def __init__(self, matchups, ev_vocab, tok_mod=4096):
        self.matchups = matchups
        self.m_index = {m: i for i, m in enumerate(matchups)}
        self.ev_vocab = ev_vocab
        self.phase_index = {p: i for i, p in enumerate(PHASES)}
        self.et_index = {t: i for i, t in enumerate(EVENT_TYPES)}
        self.tok_mod = tok_mod  # fold 65536 card-id hashes into a smaller embed table

    @property
    def scalar_dim(self):
        return len(SCALAR_KEYS) + len(PHASES) + 1 + len(self.matchups)  # + my_turn

    def scalars(self, r):
        v = [float(r.get(k, 0) or 0) for k in SCALAR_KEYS]
        ph = [0.0] * len(PHASES); ph[self.phase_index.get(r.get("phase", ""), len(PHASES) - 1)] = 1.0
        mt = [1.0 if r.get("my_turn", False) else 0.0]
        mm = [0.0] * len(self.matchups)
        mi = self.m_index.get(r.get("matchup", "unknown"))
        if mi is not None:
            mm[mi] = 1.0
        return np.array(v + ph + mt + mm, dtype=np.float32)

    def toks(self, r):
        tok = r.get("tok", []); mask = r.get("tokmask", [])
        ids = [((t % (self.tok_mod - 1)) + 1) for t, m in zip(tok, mask) if m][:TOK_MAX]
        a = np.zeros(TOK_MAX, dtype=np.int64); m = np.zeros(TOK_MAX, dtype=np.float32)
        if ids:
            a[:len(ids)] = ids; m[:len(ids)] = 1.0
        else:
            m[0] = 1.0
        return a, m

    def events(self, r):
        evs = r.get("events", [])[-EV_MAX:]
        et = np.zeros(EV_MAX, dtype=np.int64); ec = np.zeros(EV_MAX, dtype=np.int64)
        en = np.zeros((EV_MAX, 3), dtype=np.float32); em = np.zeros(EV_MAX, dtype=np.float32)
        cur = r.get("turn", 0)
        for j, e in enumerate(evs):
            et[j] = self.et_index.get(e.get("ty", ""), self.et_index["<unk>"])
            ec[j] = self.ev_vocab.get(e.get("c", ""), 1)
            en[j, 0] = 1.0 if e.get("a", 1) == 1 else 0.0
            en[j, 1] = math.tanh(float(e.get("n", 0) or 0) / 5.0)
            en[j, 2] = math.tanh(float(cur - e.get("t", cur)) / 5.0)
            em[j] = 1.0
        if len(evs) == 0:
            em[0] = 1.0
        return et, ec, en, em


def collate(batch):
    S = torch.from_numpy(np.stack([b["s"] for b in batch]))
    tok = torch.from_numpy(np.stack([b["tok"] for b in batch]))
    tokm = torch.from_numpy(np.stack([b["tokm"] for b in batch]))
    et = torch.from_numpy(np.stack([b["et"] for b in batch]))
    ec = torch.from_numpy(np.stack([b["ec"] for b in batch]))
    en = torch.from_numpy(np.stack([b["en"] for b in batch]))
    em = torch.from_numpy(np.stack([b["em"] for b in batch]))
    y = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
    return S, tok, tokm, et, ec, en, em, y


# ---------------- models ----------------
class SnapEncoder(nn.Module):
    def __init__(self, scalar_dim, tok_mod, d=48):
        super().__init__()
        self.tok_emb = nn.Embedding(tok_mod, d, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(scalar_dim + d, 96), nn.ReLU(), nn.LayerNorm(96))

    def forward(self, S, tok, tok_mask):
        te = self.tok_emb(tok) * tok_mask.unsqueeze(-1)
        tp = te.sum(1) / tok_mask.sum(1, keepdim=True).clamp(min=1)
        return self.proj(torch.cat([S, tp], -1))


class HistEncoder(nn.Module):
    def __init__(self, ev_vocab, d=32, gru=False):
        super().__init__()
        self.card_emb = nn.Embedding(ev_vocab, d, padding_idx=0)
        self.et_emb = nn.Embedding(len(EVENT_TYPES), 12)
        in_d = d + 12 + 3
        self.gru = gru
        if gru:
            self.rnn = nn.GRU(in_d, 64, batch_first=True)
            self.out_d = 64
        else:
            self.mlp = nn.Sequential(nn.Linear(in_d, 64), nn.ReLU())
            self.out_d = 64

    def forward(self, et, ecard, enum, emask):
        x = torch.cat([self.card_emb(ecard), self.et_emb(et), enum], -1)
        if self.gru:
            lengths = emask.sum(1).clamp(min=1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, h = self.rnn(packed)
            return h[-1]
        h = self.mlp(x) * emask.unsqueeze(-1)
        return h.sum(1) / emask.sum(1, keepdim=True).clamp(min=1)


class Probe(nn.Module):
    def __init__(self, scalar_dim, tok_mod, ev_vocab, arm):
        super().__init__()
        self.arm = arm
        self.has_snap = (arm != "H")
        self.has_hist = (arm != "S")
        d = 0
        if self.has_snap:
            self.snap = SnapEncoder(scalar_dim, tok_mod); d += 96
        if self.has_hist:
            self.hist = HistEncoder(ev_vocab, gru=(arm == "S+H+GRU")); d += self.hist.out_d
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, S, tok, tok_mask, et, ecard, enum, emask):
        parts = []
        if self.has_snap:
            parts.append(self.snap(S, tok, tok_mask))
        if self.has_hist:
            parts.append(self.hist(et, ecard, enum, emask))
        z = torch.cat(parts, -1) if len(parts) > 1 else parts[0]
        return self.head(z).squeeze(-1)


# ---------------- metrics ----------------
def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(p) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ce(y, p):
    y = np.asarray(y); p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="E:/mage-training/corpus/affinity/*.jsonl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="S,S+H,S+H+GRU")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    import os as _os
    torch.set_num_threads(max(1, (_os.cpu_count() or 4) // 2))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    rows = load_rows(args.glob)
    print(f"loaded {len(rows)} rows from {args.glob}")
    if len(rows) < 200:
        print("too few rows; generate more corpus first."); return
    won = Counter(r.get("won") for r in rows)
    print("won dist:", dict(won), " base rate:", round(sum(1 for r in rows if r.get('won') == 1) / len(rows), 3))
    print("slice dist:", Counter(slice_of(r) for r in rows).most_common())
    print("matchup dist:", Counter(r.get('matchup') for r in rows).most_common())

    matchups, ev_vocab = build_vocabs(rows)
    feat = Featurizer(matchups, ev_vocab)

    # game-level split (no row leakage)
    games = sorted({r["game"] for r in rows})
    random.Random(args.seed).shuffle(games)
    cut = int(len(games) * 0.8)
    train_g, val_g = set(games[:cut]), set(games[cut:])

    def prep(r):
        tok, tokm = feat.toks(r)
        et, ec, en, em = feat.events(r)
        return {"s": feat.scalars(r), "tok": tok, "tokm": tokm,
                "et": et, "ec": ec, "en": en, "em": em,
                "y": float(r.get("won", 0)), "slice": slice_of(r)}
    train = [prep(r) for r in rows if r["game"] in train_g]
    val = [prep(r) for r in rows if r["game"] in val_g]
    print(f"train rows {len(train)} ({len(train_g)} games) | val rows {len(val)} ({len(val_g)} games)")

    def run(arm):
        torch.manual_seed(args.seed)
        model = Probe(feat.scalar_dim, feat.tok_mod, len(ev_vocab), arm)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        bs = 128
        for ep in range(args.epochs):
            model.train(); random.shuffle(train)
            for i in range(0, len(train), bs):
                b = train[i:i + bs]
                S, tk, tm, et, ec, en, em, y = collate(b)
                opt.zero_grad(); out = model(S, tk, tm, et, ec, en, em)
                loss = lossf(out, y); loss.backward(); opt.step()
        # eval
        model.eval()
        with torch.no_grad():
            S, tk, tm, et, ec, en, em, y = collate(val)
            p = torch.sigmoid(model(S, tk, tm, et, ec, en, em)).numpy()
        return p

    ys = np.array([b["y"] for b in val])
    slices = np.array([b["slice"] for b in val])
    import time as _time
    preds = {}
    for arm in arms:
        t0 = _time.time()
        preds[arm] = run(arm)
        print(f"[trained {arm} in {_time.time()-t0:.0f}s]", flush=True)

    print("\n=== held-out results (game-level split) ===")
    groups = [("all", np.ones(len(val), bool))]
    for name in ["combat", "response", "main_easy"]:
        groups.append((name, slices == name))
    hdr = f"{'slice':<11}{'n':>6}{'base':>7}" + "".join(f"{a+' AUC':>12}{a+' CE':>11}" for a in arms)
    print(hdr)
    for gname, gmask in groups:
        if gmask.sum() < 8:
            print(f"{gname:<11}{int(gmask.sum()):>6}  (too few)"); continue
        yy = ys[gmask]; base = yy.mean()
        cells = ""
        for a in arms:
            pp = preds[a][gmask]
            cells += f"{auc(yy, pp):>12.3f}{ce(yy, pp):>11.3f}"
        print(f"{gname:<11}{int(gmask.sum()):>6}{base:>7.3f}{cells}")

    hist_arms = [a for a in arms if a != "S"]
    print("\n=== history gain over S (positive = history helps) ===")
    for gname, gmask in groups:
        if gmask.sum() < 8 or "S" not in arms or not hist_arms:
            continue
        yy = ys[gmask]
        aS = auc(yy, preds["S"][gmask]); cS = ce(yy, preds["S"][gmask])
        aH = max(auc(yy, preds[a][gmask]) for a in hist_arms)
        cH = min(ce(yy, preds[a][gmask]) for a in hist_arms)
        dauc = aH - aS
        dce_rel = (cS - cH) / cS * 100 if cS > 0 else 0
        flag = "  <-- PASS" if (dauc >= 0.03 or dce_rel >= 5.0) else ""
        print(f"  {gname:<11} dAUC(best H - S)={dauc:+.3f}  dCE_rel={dce_rel:+.1f}%{flag}")
    # ---- contested-state / residual analysis (Codex #68) ----
    if "S" in arms:
        pS = preds["S"]
        contested = (pS >= 0.35) & (pS <= 0.65)
        print(f"\n=== CONTESTED-STATE analysis (S uncertain: 0.35<=pS<=0.65; n={int(contested.sum())}/{len(val)}) ===")
        print("history has the most headroom here -- outcome is NOT state-determined")
        for gname, gmask in [("contested-all", contested),
                             ("contested-combat", contested & (slices == "combat")),
                             ("contested-response", contested & (slices == "response"))]:
            if gmask.sum() < 20:
                print(f"  {gname:<20} n={int(gmask.sum())}  (too few)"); continue
            yy = ys[gmask]
            row = f"  {gname:<20} n={int(gmask.sum()):>4} base={yy.mean():.3f}"
            for a in arms:
                row += f"  {a}:AUC={auc(yy, preds[a][gmask]):.3f}/CE={ce(yy, preds[a][gmask]):.3f}"
            print(row)
            hist_arms = [a for a in arms if a != "S"]
            if hist_arms:
                aS = auc(yy, pS[gmask]); cS = ce(yy, pS[gmask])
                aH = max(auc(yy, preds[a][gmask]) for a in hist_arms)
                cH = min(ce(yy, preds[a][gmask]) for a in hist_arms)
                dr = (cS - cH) / cS * 100 if cS > 0 else 0
                flag = "  <-- history helps" if (aH - aS >= 0.03 or dr >= 5) else ""
                print(f"       dAUC(bestH-S)={aH-aS:+.3f} dCE_rel={dr:+.1f}%{flag}")
        if "H" in arms:
            yy = ys[contested]
            print(f"  [H-only standalone on contested]: AUC={auc(yy, preds['H'][contested]):.3f} "
                  f"(>0.55 => history carries signal S lacks in the ambiguous region)")

    print("\nPASS bar (Codex #67/#68): terminal-outcome is a WEAK/saturated target (S~0.89).")
    print("Contested-state gain or H-only-contested AUC>0.55 => build CP7 BC-fit. Flat here is still NOT a kill.")


if __name__ == "__main__":
    main()
