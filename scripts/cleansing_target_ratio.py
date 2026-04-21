"""Scan recent Wildfire game logs for Cleansing Wildfire target decisions
and compute the self-target ratio (target own land vs opponent's land).

Usage:
    py -3.12 scripts/cleansing_target_ratio.py [--since <YYYY-MM-DDTHH:MM>] [--n <count>]
"""
import argparse
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WILDFIRE_GAMES = REPO_ROOT / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Wildfire/logs/games"


def scan_file(path: Path):
    """Return list of (selected_option_type, score) tuples for Cleansing Wildfire target decisions."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    decisions = []
    # Find each Cleansing Wildfire target decision: TARGET_PICK with Cleansing Wildfire on stack.
    # Pattern: block starting "GAME STATE: ... Cleansing Wildfire (controller=..." ending at "SELECTED: ..."
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if "Cleansing Wildfire (controller=" in lines[i]:
            # Look backward for DECISION header (TARGET_PICK)
            header_idx = -1
            for j in range(i, max(0, i - 40), -1):
                if "DECISION #" in lines[j] and "TARGET_PICK" in lines[j]:
                    header_idx = j
                    break
            if header_idx >= 0:
                # Forward: find OPTIONS section + SELECTED line
                options = []
                selected = None
                for j in range(i, min(len(lines), i + 60)):
                    m_opt = re.match(r"\s*(?:>>>)?\s*\[\d+\]\s+[\d.]+\s-\s(.+)$", lines[j])
                    if m_opt:
                        options.append(m_opt.group(1).strip())
                    m_sel = re.match(r"SELECTED:\s+(.+)$", lines[j])
                    if m_sel:
                        selected = m_sel.group(1).strip()
                        break
                if selected:
                    # Determine self vs opponent
                    if "(you)" in selected:
                        kind = "self"
                    elif "(" in selected:
                        kind = "opponent"
                    else:
                        kind = "unknown"
                    decisions.append((kind, selected))
            i += 10
        else:
            i += 1
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Last N games to scan")
    parser.add_argument("--since", type=str, default=None, help="Only files modified since (epoch timestamp)")
    args = parser.parse_args()

    files = []
    for sub in ("evaluation", "training"):
        p = WILDFIRE_GAMES / sub
        if not p.exists():
            continue
        files.extend(p.glob("*.txt"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    files = files[: args.n]

    by_kind = {"self": 0, "opponent": 0, "unknown": 0}
    total = 0
    sample_self = []
    sample_opp = []
    for f in files:
        ds = scan_file(f)
        for kind, sel in ds:
            by_kind[kind] = by_kind.get(kind, 0) + 1
            total += 1
            if kind == "self" and len(sample_self) < 3:
                sample_self.append((f.name, sel))
            elif kind == "opponent" and len(sample_opp) < 3:
                sample_opp.append((f.name, sel))

    print(f"Scanned {len(files)} most-recent Wildfire games.")
    print(f"Cleansing Wildfire target decisions: {total}")
    if total == 0:
        return
    for k in ("self", "opponent", "unknown"):
        v = by_kind.get(k, 0)
        print(f"  {k}: {v} ({100*v/total:.1f}%)")

    print("\nSelf-target examples:")
    for f, sel in sample_self:
        print(f"  {f}: {sel}")
    print("\nOpponent-target examples:")
    for f, sel in sample_opp:
        print(f"  {f}: {sel}")


if __name__ == "__main__":
    main()
