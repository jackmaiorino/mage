#!/usr/bin/env python3
"""Frozen history probe (Codex gate 1, direct falsifier for trajectory-context).

Question: does the agent's own recent ACTION HISTORY carry predictive signal about
"will it cast Balustrade Spy within N turns" that the current OBSERVABLE STATE does
not? If not, history is redundant (game state is Markov) and the trajectory-context
lever is dead -- skip the multi-day train.

Per agent decision, builds:
  STATE features (observable now): turn, our life/board/gy/hand, opp life/board.
  HISTORY features (last K decisions): counts by action class (pass/cast/land/other),
    spells cast so far this game, passes in a row, decisions so far.
  LABEL: agent casts Balustrade Spy within the next N turns from this decision.

Trains logistic probes (torch, game-clustered train/test) on STATE-only vs STATE+HISTORY,
reports test AUC + the lift, overall and on the never-cast (durdle) game slice.

Usage: python history_probe.py DIR [DIR ...] [--k 5] [--n 3]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

STATE_RE = re.compile(
    r"PlayerRL1 L(-?\d+) H(\d+).*?B(\d+).*?G(\d+).*?\|\|\s*\S+ L(-?\d+) H(\d+) B(\d+)")
TURN_RE = re.compile(r"DECISION #\d+ - Turn (\d+)")


def action_class(txt):
    t = (txt or "").strip().lower()
    if t == "pass" or t.startswith("pass"):
        return "pass"
    if "balustrade spy" in t and t.startswith("cast"):
        return "spy"
    if t.startswith("cast"):
        return "cast"
    if t.startswith("play") or "cycling" in t or "land grant" in t:
        return "land"
    return "other"


def parse_game(path):
    """Return list of decisions: dict(turn, st=[state feats], act=class). + casts list of turns."""
    decs = []
    cur_turn = 0
    pend_state = None
    spy_turns = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            mt = TURN_RE.search(s)
            if mt:
                cur_turn = int(mt.group(1))
            if "STATE:" in s:
                m = STATE_RE.search(s)
                if m:
                    pend_state = [int(m.group(1)), int(m.group(3)), int(m.group(4)),
                                  int(m.group(2)), int(m.group(5)), int(m.group(7)), cur_turn]
            elif "REPLAY_DECISION_JSON:" in s:
                try:
                    obj = json.loads(s.split("REPLAY_DECISION_JSON:", 1)[1].strip())
                except Exception:
                    continue
                if obj.get("player") != "PlayerRL1" or obj.get("action_type") == "MULLIGAN":
                    continue
                ct = obj.get("chosen_texts") or [""]
                cls = action_class(ct[0] if ct else "")
                st = pend_state if pend_state else [0, 0, 0, 0, 0, 0, cur_turn]
                decs.append(dict(turn=cur_turn, st=st, act=cls))
                if cls == "spy":
                    spy_turns.append(cur_turn)
    return decs, spy_turns


def build(dirs, K, N):
    logs = []
    for d in dirs:
        logs += glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    games = []
    for p in logs:
        decs, spy_turns = parse_game(p)
        if decs:
            games.append((decs, spy_turns))
    # rows: (state_feat, hist_feat, label, game_idx, ever_cast)
    rows = []
    for gi, (decs, spy_turns) in enumerate(games):
        ever_cast = len(spy_turns) > 0
        spells = 0
        for i, d in enumerate(decs):
            # history over last K decisions
            window = decs[max(0, i - K):i]
            cnt = {"pass": 0, "cast": 0, "land": 0, "spy": 0, "other": 0}
            for w in window:
                cnt[w["act"]] += 1
            # passes in a row ending here
            pir = 0
            for w in reversed(window):
                if w["act"] == "pass":
                    pir += 1
                else:
                    break
            hist = [cnt["pass"], cnt["cast"], cnt["land"], cnt["other"], pir, spells, i]
            if d["act"] in ("cast", "spy", "land"):
                spells += 1
            # label: a Spy cast occurs within N turns after this decision's turn
            label = 1 if any(d["turn"] < st <= d["turn"] + N for st in spy_turns) else 0
            rows.append((d["st"], hist, label, gi, ever_cast))
    return rows


def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    # simple tie handling via scipy-free average rank
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def train_probe(X, y, tr, te, epochs=300):
    Xt = torch.tensor(X[tr], dtype=torch.float32); yt = torch.tensor(y[tr], dtype=torch.float32)
    mu = Xt.mean(0, keepdim=True); sd = Xt.std(0, keepdim=True).clamp(min=1e-6)
    Xt = (Xt - mu) / sd
    Xe = (torch.tensor(X[te], dtype=torch.float32) - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); logit = Xt @ w + b; loss = lossf(logit, yt) + 1e-3 * (w * w).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        return (Xe @ w + b).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = build(args.dirs, args.k, args.n)
    St = np.array([r[0] for r in rows], dtype=np.float32)
    Hi = np.array([r[1] for r in rows], dtype=np.float32)
    y = np.array([r[2] for r in rows], dtype=np.float32)
    gi = np.array([r[3] for r in rows])
    ever = np.array([r[4] for r in rows])
    ngames = gi.max() + 1
    print(f"rows={len(rows)} games={ngames} label+rate={y.mean():.3f} K={args.k} N={args.n}")

    rng = np.random.default_rng(args.seed)
    g_order = rng.permutation(ngames)
    te_games = set(g_order[: ngames // 3].tolist())
    te = np.array([i for i in range(len(rows)) if gi[i] in te_games])
    tr = np.array([i for i in range(len(rows)) if gi[i] not in te_games])

    Xs = St
    Xsh = np.concatenate([St, Hi], axis=1)
    s_state = train_probe(Xs, y, tr, te)
    s_both = train_probe(Xsh, y, tr, te)
    yte = y[te]
    a_state = auc(s_state, yte); a_both = auc(s_both, yte)
    print(f"\nTEST AUC  state-only={a_state:.3f}   state+history={a_both:.3f}   lift={a_both - a_state:+.3f}")

    # never-cast (durdle) slice: test decisions from games that never cast Spy.
    # Label there is ~all 0, so instead measure: does history change the SCORE the probe
    # assigns (i.e., does history flag these as low-cast-prob)? Report mean predicted prob.
    te_never = te[ever[te] == False]
    te_ever = te[ever[te] == True]
    if len(te_never) and len(te_ever):
        ps_state = 1 / (1 + np.exp(-s_state)); ps_both = 1 / (1 + np.exp(-s_both))
        idx_never = np.isin(te, te_never); idx_ever = np.isin(te, te_ever)
        print(f"\nmean P(cast<=N)  EVER-cast games: state={ps_state[idx_ever].mean():.3f} both={ps_both[idx_ever].mean():.3f}")
        print(f"mean P(cast<=N)  NEVER-cast games: state={ps_state[idx_never].mean():.3f} both={ps_both[idx_never].mean():.3f}")
        print("  (history should push NEVER-cast/durdle decisions to LOWER P(cast) if it carries durdle signal)")

    print("\nGATE: lift >= ~0.03-0.05 AND concentrated in durdle slice => history carries signal -> trajectory-context worth a smoke.")
    print("      lift ~0 => history redundant with observable state (Markov) -> trajectory-context DEAD, skip the train.")


if __name__ == "__main__":
    main()
