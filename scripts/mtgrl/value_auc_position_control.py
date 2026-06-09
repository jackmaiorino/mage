#!/usr/bin/env python3
"""Confound check: does the mid-horizon WM value-AUC advantage survive controlling
for game length / absolute game position?

The headline AUC tool (value_auc_bootstrap.py) buckets decisions by DISTANCE-FROM-
TERMINAL. WM games are slightly shorter than baseline. Two confounds to rule out:

 (i)  GAME-LENGTH MIX: a given distance bucket (e.g. 21-40) is populated by
      different games / different absolute positions in WM vs baseline. The AUC
      "gain" could just reflect that "21-40 from terminal" lands at an earlier,
      easier-to-call point in WM's shorter games.
 (ii) ABSOLUTE-POSITION CONFLATION: "21-40 from terminal" = early game in a short
      won game but mid game in a long one.

This script (a) reproduces the distance-from-terminal AUC as a sanity check,
(b) recomputes AUC bucketed by ABSOLUTE TURN, (c) reports game-length distributions,
(d) re-runs the distance-bucket AUC on a LENGTH-MATCHED subset (games whose total
decision count falls in a window common to both models), and (e) reports, for each
distance bucket, the distribution of absolute turn and game-fraction the bucket's
decisions occupy in WM vs baseline (to expose conflation).

Mirrors value_myopia_probe.py parsing EXACTLY (PlayerRL1, value_score, MULLIGAN
skipped, RESULT: line). Cluster (whole-game) bootstrap for all CIs.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

RL_PLAYER = "PlayerRL1"
DIST_BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 70), (71, 110), (111, 10**9)]
DIST_LABELS = [(f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")) for lo, hi in DIST_BUCKETS]
# Absolute turn buckets (MTG turns; the agent goes off ~turn 4-8 typically).
TURN_BUCKETS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 20), (21, 10**9)]
TURN_LABELS = [(f"{lo}-{hi}" if hi < 10**9 else f"{lo}+") for lo, hi in TURN_BUCKETS]


def _label(d, buckets):
    for lo, hi in buckets:
        if lo <= d <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
    return "?"


def parse_log(path):
    """Return (outcome, [(ordinal, value, turn)]) mirroring value_myopia_probe parsing."""
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
                decs.append((int(obj["ordinal"]), float(obj["value_score"]), int(obj.get("turn", 0))))
    return outcome, decs


def rank_auc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    vals = [v for v, _ in allv]
    n = len(allv)
    rank_of = [0.0] * n
    j = 0
    while j < n:
        k = j
        while k + 1 < n and vals[k + 1] == vals[j]:
            k += 1
        avg = (j + 1 + k + 1) / 2.0
        for t in range(j, k + 1):
            rank_of[t] = avg
        j = k + 1
    rsum_pos = sum(rank_of[i] for i, (_, lab) in enumerate(allv) if lab == 1)
    npos, nneg = len(pos), len(neg)
    u = rsum_pos - npos * (npos + 1) / 2.0
    return u / (npos * nneg)


def load_games(dirs):
    """One entry per game: (outcome, n_decisions, dist_bvals{}, turn_bvals{}, raw_decs[])."""
    logs = []
    for d in dirs:
        logs += glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    games = []
    for p in logs:
        outcome, decs = parse_log(p)
        if outcome not in ("win", "loss") or len(decs) < 3:
            continue
        last = max(o for o, _, _ in decs)
        dist_bvals = defaultdict(list)
        turn_bvals = defaultdict(list)
        for ordn, v, turn in decs:
            dist_bvals[_label(last - ordn, DIST_BUCKETS)].append(v)
            turn_bvals[_label(turn, TURN_BUCKETS)].append(v)
        games.append((outcome, len(decs), dict(dist_bvals), dict(turn_bvals), decs))
    return games


def bucket_auc(games, which, labels):
    idx = 2 if which == "dist" else 3
    buck = {b: {"win": [], "loss": []} for b in labels}
    for g in games:
        outcome, bvals = g[0], g[idx]
        for b, vs in bvals.items():
            if b in buck:
                buck[b][outcome].extend(vs)
    return {b: rank_auc(buck[b]["win"], buck[b]["loss"]) for b in labels}


def cluster_counts(games, which, labels):
    idx = 2 if which == "dist" else 3
    cw = defaultdict(int)
    cl = defaultdict(int)
    for g in games:
        outcome, bvals = g[0], g[idx]
        for b in bvals:
            (cw if outcome == "win" else cl)[b] += 1
    return cw, cl


def pctl(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def diff_table(wm, base, which, labels, boot, ci, rng):
    lo_p, hi_p = (100 - ci) / 2.0, 100 - (100 - ci) / 2.0
    obs_wm = bucket_auc(wm, which, labels)
    obs_base = bucket_auc(base, which, labels)
    cw_wm, cl_wm = cluster_counts(wm, which, labels)
    cw_b, cl_b = cluster_counts(base, which, labels)
    nw, nb = len(wm), len(base)
    boot_diff = defaultdict(list)
    boot_wm = defaultdict(list)
    boot_base = defaultdict(list)
    for _ in range(boot):
        ws = [wm[i] for i in rng.integers(0, nw, nw)]
        bs = [base[i] for i in rng.integers(0, nb, nb)]
        aw = bucket_auc(ws, which, labels)
        ab = bucket_auc(bs, which, labels)
        for b in labels:
            if aw[b] is not None:
                boot_wm[b].append(aw[b])
            if ab[b] is not None:
                boot_base[b].append(ab[b])
            if aw[b] is not None and ab[b] is not None:
                boot_diff[b].append(aw[b] - ab[b])
    print(f"{'bucket':>8} | {'WMw/l':>9} | {'BASEw/l':>9} | {'WM_AUC':>7} | {'BASE_AUC':>8} | "
          f"{'diff[%dCI]' % ci:>22} | P(>0)" % ())
    print("-" * 95)
    for b in labels:
        if obs_wm[b] is None and obs_base[b] is None:
            continue
        wmc = f"{cw_wm.get(b,0)}/{cl_wm.get(b,0)}"
        bsc = f"{cw_b.get(b,0)}/{cl_b.get(b,0)}"
        wa = f"{obs_wm[b]:.3f}" if obs_wm[b] is not None else "  -"
        ba = f"{obs_base[b]:.3f}" if obs_base[b] is not None else "  -"
        if boot_diff[b]:
            d_obs = obs_wm[b] - obs_base[b]
            d_lo, d_hi = pctl(boot_diff[b], lo_p), pctl(boot_diff[b], hi_p)
            pgt = float(np.mean(np.array(boot_diff[b]) > 0))
            sig = " *" if (d_lo > 0 or d_hi < 0) else ""
            ds = f"{d_obs:+.3f}[{d_lo:+.3f},{d_hi:+.3f}]"
            print(f"{b:>8} | {wmc:>9} | {bsc:>9} | {wa:>7} | {ba:>8} | {ds:>22} | {pgt:.3f}{sig}")
        else:
            print(f"{b:>8} | {wmc:>9} | {bsc:>9} | {wa:>7} | {ba:>8} | {'n/a':>22} |")


def length_summary(games, tag):
    won = [g[1] for g in games if g[0] == "win"]
    lost = [g[1] for g in games if g[0] == "loss"]
    def stat(a):
        if not a:
            return "n=0"
        a = sorted(a)
        return (f"n={len(a)} med={np.median(a):.0f} mean={np.mean(a):.1f} "
                f"p25={np.percentile(a,25):.0f} p75={np.percentile(a,75):.0f} max={max(a)}")
    print(f"  {tag} won : {stat(won)}")
    print(f"  {tag} loss: {stat(lost)}")
    return won, lost


def conflation_report(games, tag, contested):
    """For each contested distance bucket, summarize absolute turn + game-fraction
    of the decisions that fall in it (won/loss separately)."""
    # accumulate per (bucket, outcome): list of (turn, frac_of_game)
    acc = defaultdict(lambda: defaultdict(list))
    for outcome, ndec, dist_bvals, turn_bvals, decs in games:
        last = max(o for o, _, _ in decs)
        for ordn, v, turn in decs:
            lab = _label(last - ordn, DIST_BUCKETS)
            if lab in contested:
                frac = ordn / last if last > 0 else 0.0  # 0=game start, 1=terminal
                acc[lab][outcome].append((turn, frac))
    print(f"  [{tag}] absolute turn & game-fraction of decisions inside contested distance buckets:")
    for b in contested:
        for oc in ("win", "loss"):
            rows = acc[b][oc]
            if not rows:
                continue
            turns = np.array([t for t, _ in rows])
            fracs = np.array([f for _, f in rows])
            print(f"    {b:>6} {oc:>4}: n={len(rows):>4}  turn med={np.median(turns):.0f} "
                  f"mean={turns.mean():.1f}  game-frac med={np.median(fracs):.2f} mean={fracs.mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ci", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    wm = load_games(args.wm)
    base = load_games(args.baseline)
    wm_w = sum(1 for g in wm if g[0] == "win")
    base_w = sum(1 for g in base if g[0] == "win")
    print(f"WM games={len(wm)} (won={wm_w} lost={len(wm)-wm_w}) | "
          f"BASE games={len(base)} (won={base_w} lost={len(base)-base_w}) | boot={args.boot} ci={args.ci:.0f}%")

    print("\n================ (0) GAME-LENGTH DISTRIBUTIONS (decisions/game) ================")
    wm_won, wm_lost = length_summary(wm, "WM ")
    base_won, base_lost = length_summary(base, "BASE")

    print("\n================ (1) SANITY: AUC by DISTANCE-FROM-TERMINAL (full set) ===========")
    diff_table(wm, base, "dist", DIST_LABELS, args.boot, args.ci, rng)

    print("\n================ (2) AUC by ABSOLUTE TURN NUMBER (full set) =====================")
    print("If the value head genuinely de-myopied, WM should win at EARLY absolute turns too,")
    print("not just at small distance-from-terminal that maps to late turns.")
    diff_table(wm, base, "turn", TURN_LABELS, args.boot, args.ci, rng)

    # (3) Length-matched subset: common decision-count window [overlap].
    all_lens = [g[1] for g in wm] + [g[1] for g in base]
    lo_match = max(np.percentile([g[1] for g in wm], 10), np.percentile([g[1] for g in base], 10))
    hi_match = min(np.percentile([g[1] for g in wm], 90), np.percentile([g[1] for g in base], 90))
    wm_m = [g for g in wm if lo_match <= g[1] <= hi_match]
    base_m = [g for g in base if lo_match <= g[1] <= hi_match]
    print(f"\n========= (3) LENGTH-MATCHED AUC by DISTANCE (games with {lo_match:.0f}<=n_dec<={hi_match:.0f}) =====")
    print(f"  matched: WM {len(wm_m)} games (won={sum(1 for g in wm_m if g[0]=='win')}), "
          f"BASE {len(base_m)} games (won={sum(1 for g in base_m if g[0]=='win')})")
    ml_wm_won = [g[1] for g in wm_m if g[0] == 'win']
    ml_b_won = [g[1] for g in base_m if g[0] == 'win']
    print(f"  matched mean length WM={np.mean([g[1] for g in wm_m]):.1f} BASE={np.mean([g[1] for g in base_m]):.1f}")
    diff_table(wm_m, base_m, "dist", DIST_LABELS, args.boot, args.ci, np.random.default_rng(args.seed + 1))

    # (4) Conflation: where do the contested mid buckets actually sit?
    print("\n================ (4) CONFLATION CHECK: contested mid buckets ====================")
    contested = ["11-20", "21-40", "1-2", "3-5"]
    conflation_report(wm, "WM ", contested)
    conflation_report(base, "BASE", contested)

    # (5) WIN-ONLY-vs-WIN and LOSS-ONLY value level by absolute turn (is the rank
    #     gain just that WM separates won/lost games that DIFFER in length?).
    print("\n================ (5) MEAN value_score by absolute turn (won vs lost) ============")
    for tag, games in (("WM ", wm), ("BASE", base)):
        per = defaultdict(lambda: {"win": [], "loss": []})
        for outcome, ndec, db, tb, decs in games:
            for b, vs in tb.items():
                per[b][outcome].extend(vs)
        print(f"  [{tag}]")
        for b in TURN_LABELS:
            w = per[b]["win"]; l = per[b]["loss"]
            if not w and not l:
                continue
            mw = np.mean(w) if w else float("nan")
            ml = np.mean(l) if l else float("nan")
            a = rank_auc(w, l)
            astr = f"{a:.3f}" if a is not None else "  -"
            print(f"    turn {b:>5}: nW={len(w):>4} nL={len(l):>4}  mean_win={mw:+.3f} mean_loss={ml:+.3f} AUC={astr}")


if __name__ == "__main__":
    main()
