#!/usr/bin/env python3
"""
Final temporal hedge (Codex #70 target #2): does the public event history predict
whether the OPPONENT takes an instant-speed RESPONSE on the agent's turn, beyond the
snapshot? This is tempo/intent belief (held-up mana, representing interaction) --
more genuinely sequence-dependent than hand contents.

Target (derived from corpus, no regen): for each agent decision on ITS OWN turn,
y=1 if the opponent casts/counters something between this decision and the agent's
next decision (multiset diff of consecutive rows' event windows -> strictly future,
anti-leak). Restrict to contested rows (opp has cards + untapped mana = could respond).

Arms S / S+H / H. Report AUC/logloss overall + contested + within-matchup.
PASS: S+H beats S >=0.03 AUC or >=5% rel logloss on contested reactive, holds in
multiple matchups. Flat -> close temporal.
"""
import argparse, os, sys, random
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history_probe import (load_rows, Featurizer, Probe, collate as x_collate,
                           build_vocabs, auc, ce as logloss)


def derive_response(rows):
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(r)
    out = []
    for g, rs in by_game.items():
        rs.sort(key=lambda r: r.get("seq", 0))
        for i in range(len(rs) - 1):
            r, nxt = rs[i], rs[i + 1]
            if not r.get("my_turn", False):
                continue  # agent's own turn -> opp action = instant-speed response
            def key(e):
                return (e.get("turn"), e.get("a"), e.get("ty"), e.get("c"))
            cur = Counter(key(e) for e in r.get("events", []))
            nx = Counter(key(e) for e in nxt.get("events", []))
            diff = nx - cur  # events new since this decision (strictly future)
            resp = any(a == 1 and ty in ("cast", "counter", "spellcast")
                       for (t, a, ty, c) in diff.elements())
            r = dict(r); r["_resp"] = 1.0 if resp else 0.0
            out.append(r)
    return out


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

    rows = derive_response(load_rows(args.glob))
    print(f"decisions on agent turn (with future window): {len(rows)}")
    base = np.mean([r["_resp"] for r in rows])
    print(f"opp-responds base rate: {base:.3f}")
    contested = [r for r in rows if r.get("opp_hand", 0) >= 1 and r.get("opp_untap_land", 0) >= 1]
    print(f"contested (opp has cards+mana): {len(contested)}  base {np.mean([r['_resp'] for r in contested]):.3f}")
    if len(contested) < 400:
        print("too few contested rows."); return

    matchups, ev_vocab = build_vocabs(rows)
    feat = Featurizer(matchups, ev_vocab)

    def prep(r):
        tok, tokm = feat.toks(r); et, ec, en, em = feat.events(r)
        return {"s": feat.scalars(r), "tok": tok, "tokm": tokm, "et": et, "ec": ec,
                "en": en, "em": em, "y": float(r["_resp"]),
                "matchup": r.get("matchup", "?"),
                "contested": r.get("opp_hand", 0) >= 1 and r.get("opp_untap_land", 0) >= 1}

    games = sorted({r["game"] for r in rows})
    random.Random(args.seed).shuffle(games)
    cut = int(len(games) * 0.8)
    train_g = set(games[:cut]); val_g = set(games[cut:])
    train = [prep(r) for r in rows if r["game"] in train_g]
    val = [prep(r) for r in rows if r["game"] in val_g]
    print(f"train {len(train)} | val {len(val)}")

    def run(arm):
        torch.manual_seed(args.seed)
        model = Probe(feat.scalar_dim, feat.tok_mod, len(ev_vocab), arm)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        bs = 128
        for ep in range(args.epochs):
            random.shuffle(train)
            for i in range(0, len(train), bs):
                b = train[i:i + bs]
                S, tk, tm, et, ec, en, em, y = x_collate(b)
                opt.zero_grad(); loss = lossf(model(S, tk, tm, et, ec, en, em), y)
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            S, tk, tm, et, ec, en, em, _ = x_collate(val)
            return torch.sigmoid(model(S, tk, tm, et, ec, en, em)).numpy()
    import time as _t
    preds = {}
    for a in arms:
        t0 = _t.time(); preds[a] = run(a); print(f"[trained {a} in {_t.time()-t0:.0f}s]", flush=True)

    ys = np.array([b["y"] for b in val])
    con = np.array([b["contested"] for b in val])
    mus = np.array([b["matchup"] for b in val])

    def report(name, mask):
        if mask.sum() < 30 or ys[mask].sum() < 5 or (len(ys[mask]) - ys[mask].sum()) < 5:
            print(f"  {name:<20} n={int(mask.sum())} (too few/one-class)"); return
        yy = ys[mask]
        cells = "".join(f"  {a}:AUC={auc(yy, preds[a][mask]):.3f}/LL={logloss(yy, preds[a][mask]):.3f}" for a in arms)
        print(f"  {name:<20} n={int(mask.sum()):>4} base={yy.mean():.2f}{cells}")
        if "S" in arms:
            ha = [a for a in arms if a != "S"]
            if ha:
                aS = auc(yy, preds["S"][mask]); cS = logloss(yy, preds["S"][mask])
                aH = max(auc(yy, preds[a][mask]) for a in ha); cH = min(logloss(yy, preds[a][mask]) for a in ha)
                dr = (cS - cH) / cS * 100 if cS > 0 else 0
                flag = "  <-- HISTORY HELPS" if (aH - aS >= 0.03 or dr >= 5) else ""
                print(f"       dAUC(bestH-S)={aH-aS:+.3f} dLL_rel={dr:+.1f}%{flag}")

    print("\n=== TARGET #2: predict opp instant-speed response on agent's turn ===")
    report("all", np.ones(len(val), bool))
    report("contested", con)
    print("\n=== within-matchup (contested only) ===")
    for m in sorted(set(mus)):
        report(f"matchup={m}", con & (mus == m))
    print("\nPASS (Codex #70): S+H beats S >=0.03 AUC or >=5% rel LL on contested, holds in MULTIPLE matchups.")
    print("Flat/negative -> CLOSE temporal; commit to action-quality distillation builder.")


if __name__ == "__main__":
    main()
