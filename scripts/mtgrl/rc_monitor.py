#!/usr/bin/env python3
"""Overnight monitor for the reverse-curriculum training run.

Logs RC health + combo-rate every interval; exits when the trainer JVM dies or
after a bounded number of iterations (so the parent agent is re-invoked to run
the final paired eval). Read-only except for its own monitor log.
"""
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts/mtgrl")
import eval_report as er  # noqa: E402

REPO = Path(".").resolve()
GAME_DIR = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Spy-Combo-Value/logs/games/training"
STATS = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Spy-Combo-Value/logs/stats/training_stats.csv"
GPU_LOG = "local-training/local_pbt/gpu_service.log"
OUT = "local-training/local_pbt/reverse_curriculum/rc_overnight_run/rc_monitor.log"

INTERVAL_SEC = int(os.getenv("RC_MONITOR_INTERVAL_SEC", "1200"))
ITERS = int(os.getenv("RC_MONITOR_ITERS", "21"))
RUN_START = time.time()


def java_alive() -> int:
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq java.exe"],
                             capture_output=True, text=True, timeout=30).stdout
        return out.count("java.exe")
    except Exception:
        return -1


def count_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return -1


def train_diag_count() -> int:
    try:
        with open(GPU_LOG, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for ln in f if "TRAIN_DIAG" in ln)
    except OSError:
        return -1


def recent_combo(n=60):
    """Combo-rate over the n most recently modified game logs created after run start."""
    fs = [f for f in glob.glob(f"{GAME_DIR}/*.txt") if os.path.getmtime(f) >= RUN_START - 120]
    fs.sort(key=os.path.getmtime)
    fs = fs[-n:]
    recs = [er.parse_game_log(Path(f)) for f in fs]
    recs = [r for r in recs if r.get("full_combo") is not None]
    if not recs:
        return (0, None, None, None)
    nn = len(recs)
    full = sum(1 for r in recs if r["full_combo"]) / nn
    spy = sum(1 for r in recs if r.get("has_spy")) / nn
    won = sum(1 for r in recs if r["won"]) / nn
    return (nn, round(full, 3), round(spy, 3), round(won, 3))


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stats_tail() -> str:
    try:
        with open(STATS, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-1].strip()[:90] if lines else ""
    except OSError:
        return ""


def main():
    log(f"RC monitor start: interval={INTERVAL_SEC}s iters={ITERS}")
    prev_td, prev_st = -1, -1
    dead_streak = 0
    for i in range(ITERS):
        ja = java_alive()
        td = train_diag_count()
        st_lines = count_lines(STATS)
        cn, full, spy, won = recent_combo()
        # Robust liveness: progress (train_diag or stats growing) OR java detected.
        progressed = (td > prev_td) or (st_lines > prev_st)
        alive = (ja > 0) or progressed
        prev_td, prev_st = td, st_lines
        log(f"iter={i} java={ja} alive={alive} progressed={progressed} train_diag={td} "
            f"stats_lines={st_lines} combo_n={cn} full_combo={full} spy={spy} won={won} "
            f"stats_tail=[{stats_tail()}]")
        if not alive:
            dead_streak += 1
            if dead_streak >= 2:
                log("TRAINER appears DEAD (2 consecutive no-progress + no java) -- exiting monitor.")
                break
        else:
            dead_streak = 0
        if i < ITERS - 1:
            time.sleep(INTERVAL_SEC)
    log("RC monitor done.")


if __name__ == "__main__":
    main()
