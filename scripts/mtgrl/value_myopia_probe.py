#!/usr/bin/env python3
"""Value-head myopia / discrimination gate for search-in-the-loop.

Question: does the value head separate eventual WIN from LOSS at long horizons
(i.e. during combo setup), or only near the terminal? If it only separates in the
last few decisions, a shallow value-leaf search cannot see combo progress and the
in-loop operator needs real terminal playout. If it separates 40-100 decisions out,
value-leaf search is viable.

Reads RL game logs (REPLAY_DECISION_JSON per PlayerRL1 decision carries value_score;
final 'RESULT: WIN|LOSS' line gives the label). For every decision we compute
distance-from-terminal = (last_ordinal - ordinal), then per distance bucket measure
mean value for won vs lost games and the rank AUC (P[value(won) > value(lost)]).

Usage: python value_myopia_probe.py <game_log_dir> [<game_log_dir> ...]
Thesis note: combo-progress labels here are for DIAGNOSIS only, never a train signal.
"""
import json
import sys
import glob
import os
from collections import defaultdict

RL_PLAYER = "PlayerRL1"
BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 70), (71, 110), (111, 10**9)]


def bucket_label(d):
    for lo, hi in BUCKETS:
        if lo <= d <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
    return "?"


def parse_log(path):
    """Return (outcome 'win'/'loss'/None, [(ordinal, value, action_type, gy, lib, turn, chosen)])."""
    outcome = None
    decs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("RESULT:"):
                r = s.split(":", 1)[1].strip().upper()
                outcome = "win" if r.startswith("WIN") else ("loss" if r.startswith("LOSS") else None)
            elif "REPLAY_DECISION_JSON:" in s:
                try:
                    obj = json.loads(s.split("REPLAY_DECISION_JSON:", 1)[1].strip())
                except Exception:
                    continue
                if obj.get("player") != RL_PLAYER:
                    continue
                if obj.get("action_type") == "MULLIGAN":
                    continue
                if "value_score" not in obj or obj.get("ordinal") is None:
                    continue
                decs.append((
                    int(obj["ordinal"]), float(obj["value_score"]), obj.get("action_type", ""),
                    int(obj.get("graveyard_size", 0)), int(obj.get("library_size", 0)),
                    int(obj.get("turn", 0)), " | ".join(obj.get("chosen_texts", []) or []),
                ))
    return outcome, decs


def rank_auc(pos, neg):
    """AUC = P[random pos > random neg] via Mann-Whitney, ties=0.5."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # average ranks for ties
    ranks = {}
    i = 0
    n = len(allv)
    rsum_pos = 0.0
    # assign ranks 1..n with tie averaging
    vals = [v for v, _ in allv]
    j = 0
    rank_of = [0.0] * n
    while j < n:
        k = j
        while k + 1 < n and vals[k + 1] == vals[j]:
            k += 1
        avg = (j + 1 + k + 1) / 2.0
        for t in range(j, k + 1):
            rank_of[t] = avg
        j = k + 1
    for idx, (v, lab) in enumerate(allv):
        if lab == 1:
            rsum_pos += rank_of[idx]
    npos, nneg = len(pos), len(neg)
    u = rsum_pos - npos * (npos + 1) / 2.0
    return u / (npos * nneg)


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: value_myopia_probe.py <game_log_dir> [...]")
        return
    logs = []
    for d in dirs:
        logs += glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    games = []
    for p in logs:
        outcome, decs = parse_log(p)
        if outcome in ("win", "loss") and len(decs) >= 3:
            games.append((outcome, decs))
    nwin = sum(1 for o, _ in games if o == "win")
    print(f"games parsed: {len(games)}  (won={nwin} lost={len(games)-nwin})  from {len(logs)} logs")
    if not games:
        return

    # bucket -> {'win':[v...], 'loss':[v...]}
    buck = defaultdict(lambda: {"win": [], "loss": []})
    for outcome, decs in games:
        last = max(o for o, *_ in decs)
        for ordn, v, *_ in decs:
            buck[bucket_label(last - ordn)][outcome].append(v)

    order = [(f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")) for lo, hi in BUCKETS]
    print("\n== value-head separation by distance-from-terminal (won vs lost) ==")
    print(f"{'dist':>8} {'n_win':>6} {'n_loss':>6} {'mean_win':>9} {'mean_loss':>10} {'gap':>7} {'AUC':>6}")
    for b in order:
        w = buck[b]["win"]; l = buck[b]["loss"]
        if not w and not l:
            continue
        mw = sum(w) / len(w) if w else float("nan")
        ml = sum(l) / len(l) if l else float("nan")
        auc = rank_auc(w, l)
        gap = (mw - ml) if (w and l) else float("nan")
        aucs = f"{auc:.3f}" if auc is not None else "   -"
        print(f"{b:>8} {len(w):>6} {len(l):>6} {mw:>9.3f} {ml:>10.3f} {gap:>7.3f} {aucs:>6}")

    # Combo-progress one-step probe: value at decisions that CAST Balustrade Spy (diagnosis only)
    spy_before, spy_after = [], []
    for outcome, decs in games:
        decs_sorted = sorted(decs)
        for i, (ordn, v, at, gy, lib, turn, chosen) in enumerate(decs_sorted):
            if "Balustrade Spy" in chosen and "Cast" in chosen:
                spy_before.append(v)
                if i + 1 < len(decs_sorted):
                    spy_after.append(decs_sorted[i + 1][1])
    if spy_before:
        print(f"\n== combo-progress one-step (Cast Balustrade Spy) ==")
        print(f"n={len(spy_before)}  value AT spy-cast mean={sum(spy_before)/len(spy_before):.3f}"
              + (f"  value NEXT-decision mean={sum(spy_after)/len(spy_after):.3f}" if spy_after else ""))
    else:
        print("\n(no 'Cast Balustrade Spy' decisions found in this set)")


if __name__ == "__main__":
    main()
