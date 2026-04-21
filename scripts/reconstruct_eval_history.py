#!/usr/bin/env python3
"""Reconstruct an eval_history.csv-equivalent file from raw eval game logs.

The training job's `runEvalCheckpoint` writer is silently failing to update
`eval_history.csv` mid-run (separate bug). The raw evaluation game logs
under `profiles/<PROFILE>/logs/games/evaluation/` contain a `Reason:` line
identifying the trigger episode, agent deck, and opp deck — sufficient to
reconstruct the same per-trigger aggregate.

Output: writes `eval_history_reconstructed.csv` next to the canonical file
so we don't conflict with the (broken) writer.
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

PROFILE = os.environ.get("MODEL_PROFILE", "Pauper-Standard")
REPO = Path(__file__).resolve().parent.parent
PROFILE_DIR = REPO / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "profiles" / PROFILE
LIVE_DIR = PROFILE_DIR / "logs" / "games" / "evaluation"
ARCHIVE_DIR = REPO / "local-training" / "local_pbt" / "eval_archive" / PROFILE
EVAL_DIR = ARCHIVE_DIR if ARCHIVE_DIR.exists() and any(ARCHIVE_DIR.glob("*.txt")) else LIVE_DIR
STATS_DIR = PROFILE_DIR / "logs" / "stats"
OUT_PATH = STATS_DIR / "eval_history_reconstructed.csv"


def parse_game(path: Path) -> tuple[str, str, str, str]:
    agent = opp = winner = ep = ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("Winner:"):
                    winner = line.split(":", 1)[1].strip()
                elif line.startswith("Reason:") and "EVAL-CP7" in line:
                    m_ag = re.search(r"agent=Deck - ([^,)]+)", line)
                    m_op = re.search(r"opp=Deck - ([^)]+)", line)
                    m_ep = re.search(r"ep=(\d+)", line)
                    if m_ag: agent = m_ag.group(1).strip()
                    if m_op: opp = m_op.group(1).strip()
                    if m_ep: ep = m_ep.group(1)
    except OSError:
        pass
    return ep, agent, opp, winner


def main() -> int:
    if not EVAL_DIR.exists():
        print(f"No eval dir at {EVAL_DIR}")
        return 1

    # Collect all games keyed by trigger episode.
    by_ep: dict[str, list[tuple[str, str, str]]] = defaultdict(list)  # ep -> [(agent, opp, winner)]
    for game_path in sorted(EVAL_DIR.glob("*.txt")):
        ep, agent, opp, winner = parse_game(game_path)
        if not (ep and agent and opp and winner):
            continue
        by_ep[ep].append((agent, opp, winner))

    # All distinct opp decks across all triggers, sorted, used as columns
    all_opps = sorted({op for games in by_ep.values() for (_, op, _) in games})

    rows = []
    for ep in sorted(by_ep.keys(), key=lambda e: int(e)):
        games = by_ep[ep]
        wins = sum(1 for (_, _, w) in games if w.startswith("EvalRL"))
        played = len(games)
        wr = wins / played if played else 0.0

        # Per-opp-deck winrate
        per_opp_wr = {}
        for op in all_opps:
            op_games = [(a, o, w) for (a, o, w) in games if o == op]
            if not op_games:
                per_opp_wr[op] = 0.0
                continue
            op_wins = sum(1 for (_, _, w) in op_games if w.startswith("EvalRL"))
            per_opp_wr[op] = op_wins / len(op_games)

        rows.append({
            "episode": ep,
            "wins": wins,
            "played": played,
            "winrate": f"{wr:.3f}",
            **{f"wr_Deck_-_{op.replace(' ', '_')}": f"{per_opp_wr[op]:.3f}" for op in all_opps},
        })

    if not rows:
        print("No completed eval games found.")
        return 0

    fieldnames = ["episode", "wins", "played", "winrate"] + [
        f"wr_Deck_-_{op.replace(' ', '_')}" for op in all_opps
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print()
    for row in rows[-5:]:
        print(f"  ep={row['episode']}  {row['wins']}/{row['played']} = {row['winrate']}", end="  ")
        for k, v in row.items():
            if k.startswith("wr_Deck_") and v != "0.000":
                print(f"{k.replace('wr_Deck_-_','')}={v}", end=" ")
        print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
