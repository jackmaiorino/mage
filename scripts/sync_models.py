#!/usr/bin/env python3
"""Cross-node model weight sync for distributed training.

Compares eval winrates between local and remote nodes, copies model_latest.pt
from higher-winrate node to lower-winrate node. Runs in a loop every N minutes.

Usage:
    py -3.12 scripts/sync_models.py

Env vars:
    SYNC_INTERVAL_MIN   Minutes between sync attempts (default: 10)
    REMOTE_HOST         Remote SSH host (default: haley@10.0.0.22)
    REMOTE_REPO         Remote repo path (default: C:/Users/haley/mage)
    MIN_WINRATE_GAP     Min winrate difference to trigger sync (default: 0.03)
    SYNC_PROFILES       Comma-separated profiles (default: all 4 training profiles)
"""
import csv
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_ROOT = REPO_ROOT / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "profiles"

REMOTE_HOST = os.getenv("REMOTE_HOST", "haley@10.0.0.22")
REMOTE_REPO = os.getenv("REMOTE_REPO", "C:/Users/haley/mage")
REMOTE_PROFILES_ROOT = f"{REMOTE_REPO}/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles"

SYNC_INTERVAL = float(os.getenv("SYNC_INTERVAL_MIN", "10")) * 60
MIN_GAP = float(os.getenv("MIN_WINRATE_GAP", "0.03"))
MIN_EPISODES_SINCE_SYNC = int(os.getenv("MIN_EPISODES_SINCE_SYNC", "500"))
DEFAULT_PROFILES = "Pauper-Wildfire,Pauper-Rally,Pauper-Affinity,Pauper-Elves"
PROFILES = os.getenv("SYNC_PROFILES", DEFAULT_PROFILES).split(",")

# Track episode counts at last sync to avoid syncing on stale data
_last_sync_episodes: dict = {}  # profile -> (local_ep, remote_ep)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def _read_winrate_from_file(stats_path: Path, window: int = 200) -> Tuple[int, Optional[float]]:
    """Parse episode count and rolling winrate from a training_stats.csv file."""
    if not stats_path.exists():
        return 0, None
    episode = 0
    values: deque = deque(maxlen=max(50, window))
    try:
        read_lines = max(window, 500)
        with stats_path.open("rb") as f:
            f.seek(0, 2)
            fsize = f.tell()
            seek_back = min(fsize, read_lines * 80)
            f.seek(max(0, fsize - seek_back))
            tail = f.read().decode("utf-8", errors="replace")
        for line in tail.strip().split("\n"):
            parts = line.split(",")
            if len(parts) < 5 or parts[0] == "episode":
                continue
            try:
                ep = int(parts[0])
                if ep > episode:
                    episode = ep
            except (ValueError, TypeError):
                continue
            try:
                values.append(float(parts[4]))
            except (ValueError, TypeError):
                pass
    except Exception:
        pass
    rolling = (sum(values) / len(values)) if values else None
    return episode, rolling


def read_local_winrate(profile: str) -> Tuple[int, Optional[float]]:
    stats_path = PROFILES_ROOT / profile / "logs" / "stats" / "training_stats.csv"
    return _read_winrate_from_file(stats_path)


def read_remote_winrate(profile: str) -> Tuple[int, Optional[float]]:
    """Read winrate from remote node by SCP-ing the stats file."""
    remote_stats = f"{REMOTE_PROFILES_ROOT}/{profile}/logs/stats/training_stats.csv"
    tmp = Path(tempfile.gettempdir()) / f"remote_stats_{profile}.csv"
    cmd = f'scp {REMOTE_HOST}:"{remote_stats}" "{tmp}"'
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
        if r.returncode != 0:
            return 0, None
        return _read_winrate_from_file(tmp)
    except Exception as e:
        log(f"  Error reading remote stats for {profile}: {e}")
        return 0, None
    finally:
        tmp.unlink(missing_ok=True)


def scp_local_to_remote(local_path: str, remote_path: str) -> bool:
    cmd = f'scp "{local_path}" {REMOTE_HOST}:"{remote_path}"'
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception as e:
        log(f"  SCP to remote failed: {e}")
        return False


def scp_remote_to_local(remote_path: str, local_path: str) -> bool:
    cmd = f'scp {REMOTE_HOST}:"{remote_path}" "{local_path}"'
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception as e:
        log(f"  SCP from remote failed: {e}")
        return False


def sync_profile(profile: str) -> None:
    """Compare winrates and sync model if needed."""
    local_ep, local_wr = read_local_winrate(profile)
    remote_ep, remote_wr = read_remote_winrate(profile)

    local_wr_str = f"{local_wr:.4f}" if local_wr is not None else "N/A"
    remote_wr_str = f"{remote_wr:.4f}" if remote_wr is not None else "N/A"
    log(f"  {profile}: local={local_wr_str} (ep={local_ep}) remote={remote_wr_str} (ep={remote_ep})")

    if local_wr is None or remote_wr is None:
        log(f"  Skipping: insufficient data")
        return

    if local_ep < 500 or remote_ep < 500:
        log(f"  Skipping: not enough episodes (need 500+)")
        return

    # Check that both nodes have accumulated enough NEW episodes since last sync
    prev = _last_sync_episodes.get(profile, (0, 0))
    local_delta = local_ep - prev[0]
    remote_delta = remote_ep - prev[1]
    if local_delta < MIN_EPISODES_SINCE_SYNC or remote_delta < MIN_EPISODES_SINCE_SYNC:
        log(f"  Skipping: need {MIN_EPISODES_SINCE_SYNC}+ new eps since last sync "
            f"(local_delta={local_delta}, remote_delta={remote_delta})")
        return

    gap = abs(local_wr - remote_wr)
    if gap < MIN_GAP:
        log(f"  Skipping: gap {gap:.4f} < {MIN_GAP}")
        return

    local_models = str(PROFILES_ROOT / profile / "models")
    remote_models = f"{REMOTE_PROFILES_ROOT}/{profile}/models"

    if local_wr > remote_wr:
        log(f"  LOCAL wins ({local_wr:.4f} > {remote_wr:.4f}), pushing to remote")
        ok = scp_local_to_remote(f"{local_models}/model_latest.pt", f"{remote_models}/model_latest.pt")
        if ok:
            scp_local_to_remote(f"{local_models}/model.pt", f"{remote_models}/model.pt")
            _last_sync_episodes[profile] = (local_ep, remote_ep)
            log(f"  Synced local -> remote")
        else:
            log(f"  FAILED sync local -> remote")
    else:
        log(f"  REMOTE wins ({remote_wr:.4f} > {local_wr:.4f}), pulling from remote")
        ok = scp_remote_to_local(f"{remote_models}/model_latest.pt", f"{local_models}/model_latest.pt")
        if ok:
            scp_remote_to_local(f"{remote_models}/model.pt", f"{local_models}/model.pt")
            _last_sync_episodes[profile] = (local_ep, remote_ep)
            log(f"  Synced remote -> local")
        else:
            log(f"  FAILED sync remote -> local")


def main():
    log(f"Model sync starting. Profiles: {PROFILES}")
    log(f"Remote: {REMOTE_HOST}:{REMOTE_REPO}")
    log(f"Interval: {SYNC_INTERVAL/60:.0f} min, min gap: {MIN_GAP}")

    while True:
        log("--- Sync check ---")
        for profile in PROFILES:
            profile = profile.strip()
            if not profile:
                continue
            try:
                sync_profile(profile)
            except Exception as e:
                log(f"  Error syncing {profile}: {e}")
        log(f"Next sync in {SYNC_INTERVAL/60:.0f} minutes")
        time.sleep(SYNC_INTERVAL)


if __name__ == "__main__":
    main()
