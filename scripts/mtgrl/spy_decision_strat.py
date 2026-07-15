#!/usr/bin/env python3
"""Stratified value/policy probe at Balustrade-Spy-legal decision points.

WHY (Codex Q2 disambiguation): the distance-bucketed value-AUC improved
mid-horizon (de-myopia mechanism works) but the funnel got WORSE (NO_BOARD
doubled, more overcasting). Two competing explanations:
  (A) TARGET MISS  -- the value head did NOT learn the no-board boundary, so
      it still mis-rates dominated Spy casts. Fix = switch the aux target.
  (B) CONVERSION FAIL -- the value head DID learn it (rates losing Spy-legal
      states lower) but the policy keeps casting anyway. Fix = KL/PPO dynamics.

This probe restricts to Spy-LEGAL decisions (where "Cast Balustrade Spy" is on
the menu) and reports, for WM vs BASE:
  - value-AUC at Spy-legal states (win vs loss games), game-clustered bootstrap
  - P(cast Spy | Spy-legal), split by terminal outcome (win games / loss games)
  - cast rate among Spy-legal

Read:
  - If WM value-AUC at Spy-legal states is NOT better than BASE -> (A) target miss.
  - If WM value-AUC IS better but P(cast Spy|loss-games) is HIGHER than BASE
    -> (B) the value knows but the policy overcasts -> conversion/KL problem.

Usage:
  python spy_decision_strat.py --wm DIR [DIR ...] --baseline DIR [DIR ...]
                               [--boot 2000] [--ci 90] [--spy "Balustrade Spy"]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

RL_PLAYER = "PlayerRL1"


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
    npos = len(pos)
    u = rsum_pos - npos * (npos + 1) / 2.0
    return u / (npos * len(neg))


def parse_log(path, spy_token):
    """Return (outcome, [ (value_score, p_cast_spy, chose_spy) for each Spy-legal dec ])."""
    outcome = None
    spy_decs = []
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
                if obj.get("player") != RL_PLAYER or obj.get("action_type") == "MULLIGAN":
                    continue
                if "value_score" not in obj:
                    continue
                cands = obj.get("candidate_texts") or []
                probs = obj.get("candidate_probs") or []
                # Spy-legal = a "Cast Balustrade Spy" option is present.
                spy_idx = [i for i, t in enumerate(cands)
                           if spy_token in t and t.lower().startswith("cast")]
                if not spy_idx:
                    continue
                p_spy = sum(probs[i] for i in spy_idx if i < len(probs))
                chosen = set(obj.get("chosen_indices") or [])
                chose_spy = any(i in chosen for i in spy_idx)
                spy_decs.append((float(obj["value_score"]), float(p_spy), bool(chose_spy)))
    return outcome, spy_decs


def load(dirs, spy_token):
    logs = []
    for d in dirs:
        logs += glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    games = []
    for p in logs:
        outcome, spy_decs = parse_log(p, spy_token)
        if outcome in ("win", "loss") and spy_decs:
            games.append((outcome, spy_decs))
    return games


def cohort_stats(games):
    """value-AUC at Spy-legal states + P(cast Spy) split by outcome + cast rate."""
    win_vals, loss_vals = [], []
    p_win, p_loss, casts, total = [], [], 0, 0
    for outcome, decs in games:
        for v, p, chose in decs:
            (win_vals if outcome == "win" else loss_vals).append(v)
            (p_win if outcome == "win" else p_loss).append(p)
            casts += 1 if chose else 0
            total += 1
    return {
        "auc": rank_auc(win_vals, loss_vals),
        "p_spy_win": float(np.mean(p_win)) if p_win else float("nan"),
        "p_spy_loss": float(np.mean(p_loss)) if p_loss else float("nan"),
        "cast_rate": casts / total if total else float("nan"),
        "n_decs": total,
        "n_win_decs": len(win_vals),
        "n_loss_decs": len(loss_vals),
    }


def boot_auc_diff(wm, base, boot, rng):
    nw, nb = len(wm), len(base)
    diffs, wms, bs = [], [], []
    for _ in range(boot):
        ws = [wm[i] for i in rng.integers(0, nw, nw)]
        bsm = [base[i] for i in rng.integers(0, nb, nb)]
        aw = cohort_stats(ws)["auc"]
        ab = cohort_stats(bsm)["auc"]
        if aw is not None:
            wms.append(aw)
        if ab is not None:
            bs.append(ab)
        if aw is not None and ab is not None:
            diffs.append(aw - ab)
    return diffs, wms, bs


def pctl(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ci", type=float, default=90.0)
    ap.add_argument("--spy", default="Balustrade Spy")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lo_p, hi_p = (100 - args.ci) / 2.0, 100 - (100 - args.ci) / 2.0

    wm = load(args.wm, args.spy)
    base = load(args.baseline, args.spy)
    sw, sb = cohort_stats(wm), cohort_stats(base)
    print(f"Spy-legal games: WM={len(wm)} BASE={len(base)} | spy_token='{args.spy}' boot={args.boot} ci={args.ci:.0f}%\n")

    def row(name, s):
        auc = f"{s['auc']:.3f}" if s['auc'] is not None else "n/a"
        print(f"  {name:>5}: Spy-legal value-AUC={auc}  "
              f"P(cast|win-games)={s['p_spy_win']:.3f}  P(cast|loss-games)={s['p_spy_loss']:.3f}  "
              f"cast_rate={s['cast_rate']:.3f}  (decs={s['n_decs']}, win/loss={s['n_win_decs']}/{s['n_loss_decs']})")
    row("WM", sw)
    row("BASE", sb)

    diffs, _, _ = boot_auc_diff(wm, base, args.boot, rng)
    if diffs:
        d_obs = sw['auc'] - sb['auc']
        d_lo, d_hi = pctl(diffs, lo_p), pctl(diffs, hi_p)
        pgt = float(np.mean(np.array(diffs) > 0))
        sig = " *" if (d_lo > 0 or d_hi < 0) else ""
        print(f"\n  Spy-legal value-AUC diff (WM-BASE) = {d_obs:+.3f} [{d_lo:+.3f},{d_hi:+.3f}]  P(diff>0)={pgt:.3f}{sig}")

    print("\nREAD:")
    print("  (A) target miss  : WM value-AUC NOT > BASE  -> value head didn't learn the boundary -> switch aux target.")
    print("  (B) conversion   : WM value-AUC > BASE AND P(cast|loss-games) >= BASE -> value knows, policy overcasts -> KL/PPO.")


if __name__ == "__main__":
    main()
