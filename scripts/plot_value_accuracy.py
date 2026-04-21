#!/usr/bin/env python3
"""Plot value-head accuracy trend (ASCII, no matplotlib dependency).

Usage:
    py -3.12 scripts/plot_value_accuracy.py [profile]
Default profile: Pauper-Standard-Wide
"""
import csv
import os
import sys
from pathlib import Path

PROFILE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_PROFILE", "Pauper-Standard-Wide")
REPO = Path(__file__).resolve().parent.parent
PATH = REPO / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "profiles" / PROFILE / "logs" / "stats" / "value_accuracy.csv"
THRESHOLD = float(os.environ.get("VALUE_ACCURACY_MCTS_THRESHOLD", "0.70"))


def main() -> int:
    if not PATH.exists():
        print(f"No value_accuracy.csv at {PATH}")
        return 1

    episodes = []
    accs = []
    with PATH.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ep = int(row["episode"])
                va = float(row["value_accuracy"])
                episodes.append(ep)
                accs.append(va)
            except (ValueError, KeyError):
                continue

    if not episodes:
        print("No data points yet")
        return 0

    print(f"=== Value accuracy trend ({PROFILE}) ===")
    print(f"    threshold for MCTS: {THRESHOLD:.0%}")
    print(f"    samples: {len(episodes)}   first ep: {episodes[0]}   last ep: {episodes[-1]}")
    print(f"    first accuracy: {accs[0]:.1%}   last: {accs[-1]:.1%}   max: {max(accs):.1%}")
    print()

    # Simple ASCII chart: x = sample index (compressed), y = accuracy 0-100% in 20 rows
    WIDTH = min(80, len(episodes))
    step = max(1, len(episodes) // WIDTH)
    sampled = [(episodes[i], accs[i]) for i in range(0, len(episodes), step)][:WIDTH]

    ROWS = 20
    for r in range(ROWS, -1, -1):
        y_hi = r / ROWS
        y_lo = (r - 1) / ROWS
        line = f"  {y_hi*100:5.1f}% |"
        for (_, a) in sampled:
            line += "#" if (y_lo < a <= y_hi) else (" " if a < y_lo else "|")
        # Mark threshold
        if abs(y_hi - THRESHOLD) < 0.5 / ROWS:
            line += "  <-- MCTS threshold"
        print(line)
    print("        +" + "-" * len(sampled))
    print(f"         first ep {sampled[0][0]}...last ep {sampled[-1][0]} ({len(sampled)} samples)")
    print()
    # Recent average
    recent = accs[-10:] if len(accs) >= 10 else accs
    print(f"    recent avg (last {len(recent)} samples): {sum(recent)/len(recent):.1%}")
    if accs[-1] >= THRESHOLD:
        print(f"    *** Threshold CROSSED — ready for MCTS eval ***")
    else:
        gap = THRESHOLD - accs[-1]
        print(f"    gap to threshold: {gap:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
