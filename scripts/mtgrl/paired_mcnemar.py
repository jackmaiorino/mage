"""Independent paired McNemar between two eval runs, matched by replay seed.

Each game log embeds REPLAY_OPPONENT_DECISION_JSON with a per-game "scenario"
and "seed" (derived from --replay-seed-base, so identical scenario => identical
shuffle across runs). Pair games by (scenario, seed), then McNemar on the
discordant pairs. This is the clean causal test: same shuffle, search ON vs OFF.

Usage: python paired_mcnemar.py <runA_game_logs_dir> <runB_game_logs_dir>
"""
import json
import os
import re
import sys
from math import comb

SEED = re.compile(r'REPLAY_OPPONENT_DECISION_JSON:\s*(\{.*?\})\s*$', re.MULTILINE)


def outcomes(d):
    """(scenario,seed) -> win(bool) for each game log in dir d."""
    out = {}
    for f in os.listdir(d):
        if not f.endswith(".txt"):
            continue
        with open(os.path.join(d, f), encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        win = "RESULT: WIN" in txt
        key = None
        m = SEED.search(txt)
        if m:
            try:
                j = json.loads(m.group(1))
                key = (j.get("scenario"), j.get("seed"))
            except Exception:
                key = None
        if key and key[0] is not None and key[1] is not None:
            out[key] = win
    return out


def main():
    a = outcomes(sys.argv[1])  # candidate (search ON)
    b = outcomes(sys.argv[2])  # source (search OFF)
    keys = sorted(set(a) & set(b))
    print(f"runA games={len(a)} runB games={len(b)} paired(identical shuffle)={len(keys)}")
    if not keys:
        print("no paired games")
        return
    a_w = sum(a[k] for k in keys)
    b_w = sum(b[k] for k in keys)
    print(f"on paired set: A(search ON) {a_w}/{len(keys)}={a_w/len(keys):.3f}  "
          f"B(search OFF) {b_w}/{len(keys)}={b_w/len(keys):.3f}")
    # McNemar discordant: A win & B loss (b01) vs A loss & B win (b10)
    a_win_b_loss = [k for k in keys if a[k] and not b[k]]
    a_loss_b_win = [k for k in keys if not a[k] and b[k]]
    n01, n10 = len(a_win_b_loss), len(a_loss_b_win)
    concord = len(keys) - n01 - n10
    print(f"concordant={concord}  A-win/B-loss={n01}  A-loss/B-win={n10}")
    n = n01 + n10
    if n == 0:
        print("McNemar: 0 discordant pairs -> identical outcomes on every paired game. p=1.0")
    else:
        # exact two-sided binomial p on discordant pairs vs 0.5
        k = min(n01, n10)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))
        print(f"McNemar exact two-sided p={p:.4f} (discordant n={n})")
    if a_win_b_loss:
        print("  search ON won, OFF lost:", [f"sc{k[0]}" for k in a_win_b_loss])
    if a_loss_b_win:
        print("  search ON lost, OFF won (search REGRESSIONS):", [f"sc{k[0]}" for k in a_loss_b_win])


if __name__ == "__main__":
    main()
