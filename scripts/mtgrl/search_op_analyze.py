#!/usr/bin/env python3
"""Analyze [SEARCH_OP] activation logs to measure the in-loop operator's signal quality.

The key viability metric is the DISCRIMINATING fraction: activations where >=2 candidates
reached a terminal AND their win-rates differ enough (return-gap >= min_gap) to produce a
non-degenerate policy-distillation target. Vs CP7 from natural starts these are ~0
(all playouts same outcome); self-play / combo-reaching generation should lift them.

Parses lines like:
  [SEARCH_OP] act=2 observed=2 best=1 orig=2 playouts=6 terminals=2 wallMs=7975 wr=[NaN, 1.0, ...]

Usage: python search_op_analyze.py <game_log_dir_or_glob> [min_return_gap=0.25]
"""
import sys
import glob
import os
import re
import statistics

LINE = re.compile(r"\[SEARCH_OP\]\s+act=(\d+)\s+observed=(\d+)\s+best=(-?\d+)\s+orig=(\d+)"
                  r"\s+playouts=(\d+)\s+terminals=(\d+)\s+wallMs=(\d+)\s+wr=\[([^\]]*)\]")


def parse_wr(s):
    out = []
    for tok in s.split(","):
        t = tok.strip()
        if t == "" or t.lower() == "nan":
            continue
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: search_op_analyze.py <dir_or_glob> [min_return_gap]")
        return
    target = sys.argv[1]
    min_gap = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25  # return-space gap
    if os.path.isdir(target):
        files = glob.glob(os.path.join(target, "**", "*.txt"), recursive=True)
    else:
        files = glob.glob(target, recursive=True)
    acts = 0
    obs2 = 0          # >=2 candidates reached terminal
    discriminating = 0  # >=2 observed AND return-gap >= min_gap
    overrides = 0
    walls = []
    terminals_hit = 0
    playouts_total = 0
    terminals_total = 0
    wr_gaps = []
    all_wr = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = LINE.search(line)
                    if not m:
                        continue
                    acts += 1
                    observed = int(m.group(2))
                    best = int(m.group(3)); orig = int(m.group(4))
                    playouts = int(m.group(5)); terminals = int(m.group(6))
                    walls.append(int(m.group(7)))
                    playouts_total += playouts
                    terminals_total += terminals
                    if terminals > 0:
                        terminals_hit += 1
                    if best >= 0 and best != orig:
                        overrides += 1
                    wr = parse_wr(m.group(8))
                    all_wr.extend(wr)
                    if len(wr) >= 2:
                        obs2 += 1
                        # return-space gap = 2 * (winrate gap)
                        rgap = (max(wr) - min(wr)) * 2.0
                        wr_gaps.append(rgap)
                        if rgap >= min_gap:
                            discriminating += 1
        except OSError:
            continue
    print(f"files_scanned={len(files)}  activations={acts}")
    if acts == 0:
        print("(no [SEARCH_OP] lines found)")
        return
    print(f"activations_with_any_terminal={terminals_hit} ({100*terminals_hit//acts}%)")
    print(f"playouts_total={playouts_total} terminals_total={terminals_total} "
          f"terminal_rate={terminals_total/max(1,playouts_total):.2f}")
    print(f"observed>=2 (>=2 candidates terminal) = {obs2} ({100*obs2//acts}% of activations)")
    print(f"DISCRIMINATING (obs>=2 & return_gap>={min_gap}) = {discriminating} "
          f"({100*discriminating//max(1,acts)}% of activations, {100*discriminating//max(1,obs2) if obs2 else 0}% of obs>=2)")
    print(f"overrides (best!=orig) = {overrides}")
    if walls:
        print(f"wallMs/search: median={int(statistics.median(walls))} mean={int(statistics.mean(walls))} "
              f"max={max(walls)}")
    if wr_gaps:
        nonzero = [g for g in wr_gaps if g > 0]
        print(f"return_gap among obs>=2: median={statistics.median(wr_gaps):.2f} "
              f"nonzero_gaps={len(nonzero)}/{len(wr_gaps)}")
    # Per-candidate win-rate VALUE distribution (informativeness for value-head training).
    # Saturated (all ~1.0 from passive self-play, or all ~0.0 from lethal opp) => low info.
    # A spread => informative terminal-grounded value targets.
    if all_wr:
        n = len(all_wr)
        z = sum(1 for v in all_wr if v == 0.0)
        one = sum(1 for v in all_wr if v == 1.0)
        mid = n - z - one
        print(f"per-candidate win-rate values (n={n}): ==0.0: {z} ({100*z//n}%)  "
              f"mid(0,1): {mid} ({100*mid//n}%)  ==1.0: {one} ({100*one//n}%)")
        print(f"  -> {'SATURATED (low info for value training)' if (z+one)/n > 0.9 else 'MIXED (informative)'}")


if __name__ == "__main__":
    main()
