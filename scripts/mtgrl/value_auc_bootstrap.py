#!/usr/bin/env python3
"""Game-clustered bootstrap CIs for value-head AUC by distance-from-terminal.

WHY: value_myopia_probe.py pools per-DECISION value scores and reports a point
AUC with no CI. But decisions within one game share the outcome and are highly
correlated, so the effective sample size is the number of GAMES (~tens), not
decisions (~hundreds). A naive/decision-level CI is far too tight -- which is
why a 32-game AUC flipped sign between seeds. This tool resamples whole GAMES
(cluster bootstrap) to get honest CIs, and bootstraps the WM-vs-baseline AUC
DIFFERENCE per bucket so the head-to-head is a statistical statement.

Usage:
  python value_auc_bootstrap.py --wm DIR [DIR ...] --baseline DIR [DIR ...]
                                [--boot 2000] [--ci 90]
Mirrors value_myopia_probe.py parsing (PlayerRL1, REPLAY_DECISION_JSON,
value_score, MULLIGAN skipped, RESULT: line) and bucketing EXACTLY.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

RL_PLAYER = "PlayerRL1"
BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 70), (71, 110), (111, 10**9)]
LABELS = [(f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")) for lo, hi in BUCKETS]


def bucket_label(d):
    for lo, hi in BUCKETS:
        if lo <= d <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
    return "?"


def parse_log(path):
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
                decs.append((int(obj["ordinal"]), float(obj["value_score"])))
    return outcome, decs


def rank_auc(pos, neg):
    """AUC = P[random pos > random neg], ties=0.5 (Mann-Whitney)."""
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
    """Return list of (outcome, {bucket: [values...]}) -- one entry per game."""
    logs = []
    for d in dirs:
        logs += glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    games = []
    for p in logs:
        outcome, decs = parse_log(p)
        if outcome in ("win", "loss") and len(decs) >= 3:
            last = max(o for o, _ in decs)
            bvals = defaultdict(list)
            for ordn, v in decs:
                bvals[bucket_label(last - ordn)].append(v)
            games.append((outcome, dict(bvals)))
    return games


def bucket_auc(games):
    buck = {b: {"win": [], "loss": []} for b in LABELS}
    for outcome, bvals in games:
        for b, vs in bvals.items():
            buck[b][outcome].extend(vs)
    return {b: rank_auc(buck[b]["win"], buck[b]["loss"]) for b in LABELS}


def cluster_counts(games):
    """Per bucket: (#won games, #lost games) that have >=1 decision in the bucket."""
    cw = defaultdict(int)
    cl = defaultdict(int)
    for outcome, bvals in games:
        for b in bvals:
            if outcome == "win":
                cw[b] += 1
            else:
                cl[b] += 1
    return cw, cl


def pctl(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ci", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lo_p, hi_p = (100 - args.ci) / 2.0, 100 - (100 - args.ci) / 2.0

    wm = load_games(args.wm)
    base = load_games(args.baseline)
    wm_w = sum(1 for o, _ in wm if o == "win")
    base_w = sum(1 for o, _ in base if o == "win")
    print(f"WM games={len(wm)} (won={wm_w} lost={len(wm)-wm_w}) | "
          f"BASE games={len(base)} (won={base_w} lost={len(base)-base_w}) | "
          f"boot={args.boot} ci={args.ci:.0f}%")

    obs_wm = bucket_auc(wm)
    obs_base = bucket_auc(base)
    cw_wm, cl_wm = cluster_counts(wm)
    cw_b, cl_b = cluster_counts(base)

    # Cluster bootstrap (resample whole games with replacement).
    nw, nb = len(wm), len(base)
    boot_wm = defaultdict(list)
    boot_base = defaultdict(list)
    boot_diff = defaultdict(list)
    for _ in range(args.boot):
        ws = [wm[i] for i in rng.integers(0, nw, nw)]
        bs = [base[i] for i in rng.integers(0, nb, nb)]
        aw = bucket_auc(ws)
        ab = bucket_auc(bs)
        for b in LABELS:
            if aw[b] is not None:
                boot_wm[b].append(aw[b])
            if ab[b] is not None:
                boot_base[b].append(ab[b])
            if aw[b] is not None and ab[b] is not None:
                boot_diff[b].append(aw[b] - ab[b])

    print(f"\n{'dist':>7} | {'WM_won/lost_games':>17} | {'WM AUC [CI]':>22} | "
          f"{'BASE AUC [CI]':>22} | {'WM-BASE diff [CI]':>24} | P(diff>0)")
    print("-" * 130)
    for b in LABELS:
        if obs_wm[b] is None and obs_base[b] is None:
            continue
        wmc = f"{cw_wm.get(b,0)}/{cl_wm.get(b,0)}"
        wm_s = (f"{obs_wm[b]:.3f} [{pctl(boot_wm[b],lo_p):.3f},{pctl(boot_wm[b],hi_p):.3f}]"
                if obs_wm[b] is not None else "n/a")
        bs_s = (f"{obs_base[b]:.3f} [{pctl(boot_base[b],lo_p):.3f},{pctl(boot_base[b],hi_p):.3f}]"
                if obs_base[b] is not None else "n/a")
        if boot_diff[b]:
            d_obs = (obs_wm[b] - obs_base[b]) if (obs_wm[b] is not None and obs_base[b] is not None) else float("nan")
            d_lo, d_hi = pctl(boot_diff[b], lo_p), pctl(boot_diff[b], hi_p)
            pgt = float(np.mean(np.array(boot_diff[b]) > 0))
            d_s = f"{d_obs:+.3f} [{d_lo:+.3f},{d_hi:+.3f}]"
            sig = "  *" if (d_lo > 0 or d_hi < 0) else ""
            print(f"{b:>7} | {wmc:>17} | {wm_s:>22} | {bs_s:>22} | {d_s:>24} | {pgt:.3f}{sig}")
        else:
            print(f"{b:>7} | {wmc:>17} | {wm_s:>22} | {bs_s:>22} | {'n/a':>24} |")
    print("\n* = diff CI excludes 0 (statistically distinguishable at this CI). "
          "P(diff>0) = bootstrap fraction favoring WM.")


if __name__ == "__main__":
    main()
