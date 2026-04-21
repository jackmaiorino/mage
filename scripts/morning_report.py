#!/usr/bin/env python3
"""Overnight training progress summary for Pauper-Standard.

Reads stats CSVs, produces a concise snapshot of training progress and
eval winrate trend. Safe to run alongside a live training job.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import sys
from pathlib import Path

PROFILE = os.environ.get("MODEL_PROFILE", "Pauper-Standard")
REPO = Path(__file__).resolve().parent.parent
STATS_DIR = REPO / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "profiles" / PROFILE / "logs" / "stats"
RUN_LOG = REPO / "local-training" / "local_pbt" / "run33_overnight_baseline.log"


def tail(path: Path, n: int = 20) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return [ln.rstrip("\n") for ln in lines[-n:]]


def read_eval_history() -> list[dict]:
    path = STATS_DIR / "eval_history.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_training_stats_tail(n: int = 2000) -> list[list[str]]:
    path = STATS_DIR / "training_stats.csv"
    if not path.exists():
        return []
    rows = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [r.split(",") for r in rows[-n:]]


def read_training_losses_tail(n: int = 20) -> list[list[str]]:
    path = STATS_DIR / "training_losses.csv"
    if not path.exists():
        return []
    rows = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return rows[-n:]


def run_eval_deck_tally() -> None:
    """Invoke the eval_deck_tally.py script inline and print its output."""
    import subprocess
    py_exec = sys.executable
    try:
        res = subprocess.run(
            [py_exec, str(REPO / "scripts" / "eval_deck_tally.py")],
            cwd=str(REPO),
            capture_output=True, text=True, timeout=30,
        )
        if res.stdout:
            print(res.stdout)
    except Exception as e:
        print(f"(eval_deck_tally failed: {e})")


def main() -> int:
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== Overnight report — {PROFILE} — {now} ===\n")

    # 1. Training throughput
    tail_lines = tail(RUN_LOG, 6)
    if tail_lines:
        print("Recent orchestrator heartbeats:")
        for ln in tail_lines:
            print(f"  {ln}")
        print()

    # 2. Eval history trend
    eval_hist = read_eval_history()
    if eval_hist:
        print("Eval winrate by episode (vs CP7):")
        header_printed = False
        for row in eval_hist[-15:]:
            episode = row.get("episode", "?")
            wr = row.get("winrate", "?")
            wr_rally = row.get("wr_Deck_-_Mono_Red_Rally", "")
            wr_affinity = row.get("wr_Deck_-_Grixis_Affinity", "")
            wr_elves = row.get("wr_Deck_-_Elves", "")
            wr_wildfire = row.get("wr_Deck_-_Jund_Wildfire", "")
            if not header_printed:
                print(f"  {'episode':>10} {'overall':>8} {'rally':>6} {'affinity':>9} {'elves':>6} {'wildfire':>9}")
                header_printed = True
            print(f"  {episode:>10} {wr:>8} {wr_rally:>6} {wr_affinity:>9} {wr_elves:>6} {wr_wildfire:>9}")
        print()

    # 3. Training stats: episode count and recent selfplay results
    stats_rows = read_training_stats_tail(200)
    if stats_rows:
        episodes_logged = [int(r[0]) for r in stats_rows if r and r[0].isdigit()]
        if episodes_logged:
            print(f"Stats: {len(stats_rows)} recent entries, max episode ID = {max(episodes_logged)}")

    # 4. Losses tail (to show training signal is live)
    losses_tail = read_training_losses_tail(10)
    if losses_tail:
        print("\nTraining losses (last 10):")
        for ln in losses_tail:
            print(f"  {ln}")

    # 5. Per-archetype eval winrate (from raw game logs since eval_history is broken)
    print()
    run_eval_deck_tally()

    return 0


if __name__ == "__main__":
    sys.exit(main())
