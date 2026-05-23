#!/usr/bin/env python3
"""Compute simple event/outcome attribution from exported trajectory JSONL.

This is a cheap sanity check before training a synthetic-return model. It asks:
when a decision flag appears in a game, how often did PlayerRL1 win compared
with the baseline winrate?
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def iter_games(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Trajectory JSONL path. Can be repeated.")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--min-games", type=int, default=1)
    args = parser.parse_args()

    paths = [Path(p) for p in args.input]
    games = list(iter_games(paths))
    total = len(games)
    wins = sum(1 for g in games if g.get("rl_won"))
    baseline = wins / total if total else 0.0

    event_games: Dict[str, set] = defaultdict(set)
    event_win_games: Dict[str, set] = defaultdict(set)
    event_decisions: Counter = Counter()
    selected_actions: Dict[Tuple[str, bool], Counter] = defaultdict(Counter)

    for gi, game in enumerate(games):
        won = bool(game.get("rl_won"))
        for decision in game.get("decisions", []):
            selected = str(decision.get("selected") or "")
            if selected:
                selected_actions[(decision.get("kind", "decision"), won)][selected] += 1
            for flag, value in (decision.get("flags") or {}).items():
                if not value:
                    continue
                event_games[flag].add(gi)
                event_decisions[flag] += 1
                if won:
                    event_win_games[flag].add(gi)

    rows = []
    for event, game_ids in event_games.items():
        n = len(game_ids)
        if n < args.min_games:
            continue
        w = len(event_win_games[event])
        wr = w / n if n else 0.0
        rows.append({
            "event": event,
            "games": n,
            "wins": w,
            "winrate": wr,
            "baseline_winrate": baseline,
            "lift": wr - baseline,
            "decisions": event_decisions[event],
        })
    rows.sort(key=lambda r: (r["lift"], r["games"]), reverse=True)

    print(f"games={total} wins={wins} baseline_winrate={baseline:.4f}")
    for row in rows:
        print(
            f"{row['event']},{row['wins']}/{row['games']},"
            f"wr={row['winrate']:.4f},lift={row['lift']:+.4f},decisions={row['decisions']}"
        )

    print("\nTop selected actions in wins:")
    for (kind, won), counter in sorted(selected_actions.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if not won:
            continue
        top = "; ".join(f"{name}={count}" for name, count in counter.most_common(12))
        print(f"{kind}: {top}")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "event", "games", "wins", "winrate", "baseline_winrate", "lift", "decisions"
            ])
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
