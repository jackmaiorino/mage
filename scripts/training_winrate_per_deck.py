#!/usr/bin/env python3
"""Rolling per-deck training winrate from training_cells.csv.

Usage:
    py -3.12 scripts/training_winrate_per_deck.py [profile]
Default profile: Pauper-Standard-Wide
"""
import csv
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

PROFILE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_PROFILE", "Pauper-Standard-Wide")
WINDOW = int(os.environ.get("WINRATE_WINDOW", "100"))
REPO = Path(__file__).resolve().parent.parent
PATH = REPO / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "profiles" / PROFILE / "logs" / "stats" / "training_cells.csv"


def main() -> int:
    if not PATH.exists():
        print(f"No training_cells.csv at {PATH}")
        print("This CSV is written by RLTrainer; restart training to begin populating it.")
        return 1

    cumulative: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [wins, total]
    rolling: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
    rolling_wins: dict[str, int] = defaultdict(int)
    latest_ep = 0

    with PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["episode"])
            except (ValueError, KeyError):
                continue
            latest_ep = max(latest_ep, ep)
            deck = row["rl_deck"]
            reward = float(row["final_reward"])
            won = 1 if reward > 0 else 0
            cumulative[deck][0] += won
            cumulative[deck][1] += 1
            # Rolling window
            q = rolling[deck]
            if len(q) == WINDOW:
                if q[0]:
                    rolling_wins[deck] -= 1
            q.append(won)
            if won:
                rolling_wins[deck] += 1

    print(f"=== Training winrate per RL-piloted deck (profile={PROFILE}, latest ep={latest_ep}) ===\n")
    print(f"  {'deck':>25} {'cum wins':>10} {'cum played':>11} {'cum winrate':>12} {'rolling (last '+str(WINDOW)+')':>22}")
    for deck in sorted(cumulative.keys()):
        w, t = cumulative[deck]
        cum_wr = w / t if t else 0
        r = rolling[deck]
        r_wr = rolling_wins[deck] / len(r) if r else 0
        print(f"  {deck:>25} {w:>10} {t:>11} {cum_wr:>12.3f} {r_wr:>22.3f}  (n={len(r)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
