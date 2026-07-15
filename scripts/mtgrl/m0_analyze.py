#!/usr/bin/env python3
"""M0 Forced Mulligan Counterfactual analysis (Codex #35).

Join the force-KEEP and force-MULL runs by opening hand (same seed -> same opening 7).
For each opening hand: natural choice (P_keep vs P_mull, logged even when forced),
keep-branch terminal outcome, mull-branch terminal outcome. Then:
  natural winrate  = following the policy's actual keep/mull choice
  opposite winrate = forcing the other choice
Codex decision rule (aggro slice): opposite beats natural by >=10pp -> mulligan is a
training target; <6pp -> not the bottleneck. Also report the oracle (best-branch) ceiling
and the mulligan mistake rate.

Usage: python m0_analyze.py local-training/m0_keep local-training/m0_mull
"""
import re, glob, sys, os, collections

def parse_game(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    # first mulligan decision (mulligansTaken=0): natural P_keep/P_mull + hand
    m = re.search(r"MULLIGAN_DECISION:.*?mulligansTaken=0 .*?P_keep=([0-9.]+) P_mull=([0-9.]+) hand=\[([^\]]*)\]", txt)
    if not m:
        return None
    pkeep, pmull, hand = float(m.group(1)), float(m.group(2)), m.group(3).strip()
    mo = re.search(r"RESULT:\s*(WIN|LOSS)", txt)
    if not mo:
        return None
    won = 1 if mo.group(1) == "WIN" else 0
    return {"hand": hand, "pkeep": pkeep, "pmull": pmull, "won": won}

def load(d):
    by_hand = collections.defaultdict(list)
    files = glob.glob(os.path.join(d, "*.txt")) + glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    for f in sorted(set(files)):
        g = parse_game(f)
        if g:
            by_hand[g["hand"]].append(g)
    return by_hand

def main():
    keep_d, mull_d = sys.argv[1], sys.argv[2]
    K, M = load(keep_d), load(mull_d)
    hands = set(K) & set(M)
    nat_w = opp_w = oracle_w = keep_w = mull_w = 0
    mistakes = 0; pivotal = 0; n = 0; nat_keep = 0
    for h in hands:
        # 1:1 match per hand; if duplicates, zip in order
        for gk, gm in zip(K[h], M[h]):
            n += 1
            ko, mo_ = gk["won"], gm["won"]
            keep_w += ko; mull_w += mo_
            natural_keep = gk["pkeep"] >= gk["pmull"]
            if natural_keep: nat_keep += 1
            nat = ko if natural_keep else mo_
            opp = mo_ if natural_keep else ko
            nat_w += nat; opp_w += opp
            oracle_w += max(ko, mo_)
            if ko != mo_:
                pivotal += 1
                better_keep = ko > mo_
                if better_keep != natural_keep:
                    mistakes += 1
    if n == 0:
        print(f"NO MATCHED HANDS (keep games={sum(len(v) for v in K.values())} mull games={sum(len(v) for v in M.values())} -- check logging/join)")
        return
    p = lambda x: f"{100.0*x/n:.1f}%"
    print(f"matched opening hands (keep/mull pairs) = {n}")
    print(f"  natural-choice keep rate = {p(nat_keep)}")
    print(f"force-KEEP winrate  = {p(keep_w)}")
    print(f"force-MULL winrate  = {p(mull_w)}")
    print(f"NATURAL (policy) winrate  = {p(nat_w)}")
    print(f"OPPOSITE (forced flip) winrate = {p(opp_w)}")
    delta = 100.0*(opp_w - nat_w)/n
    print(f"--- OPPOSITE - NATURAL = {delta:+.1f}pp  (Codex rule: >=10pp aggro -> mulligan IS a training target; <6pp -> not the bottleneck) ---")
    print(f"ORACLE (best-branch ceiling) winrate = {p(oracle_w)}  -> headroom over natural = {100.0*(oracle_w-nat_w)/n:+.1f}pp")
    print(f"pivotal hands (keep != mull outcome) = {pivotal} ({p(pivotal)})")
    if pivotal:
        print(f"  mulligan MISTAKES (policy chose the worse branch on a pivotal hand) = {mistakes}/{pivotal} ({100.0*mistakes/pivotal:.0f}% of pivotal)")

if __name__ == "__main__":
    main()
