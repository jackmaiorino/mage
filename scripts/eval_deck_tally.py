#!/usr/bin/env python3
"""Tally eval games by detected RL agent deck archetype and winner.

Since eval_history.csv is not being updated by the running training job,
this script reconstructs per-archetype winrate from the raw evaluation
game logs written under profiles/<PROFILE>/logs/games/evaluation/.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from collections import defaultdict

PROFILE = os.environ.get("MODEL_PROFILE", "Pauper-Standard")
ARCHIVE_DIR = Path(f"local-training/local_pbt/eval_archive/{PROFILE}")
LIVE_DIR = Path(
    f"Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/{PROFILE}/logs/games/evaluation"
)
# Prefer the archive (cumulative, never rotated) when present.
EVAL_DIR = ARCHIVE_DIR if ARCHIVE_DIR.exists() and any(ARCHIVE_DIR.glob("*.txt")) else LIVE_DIR

# Signature cards (unique per archetype)
ARCHETYPE_SIGS = {
    "Rally": {"Rally at the Hornburg", "Goblin Bushwhacker", "Burning-Tree Emissary",
              "Voldaren Epicure", "Clockwork Percussionist", "Goblin Tomb Raider"},
    "Elves": {"Elvish Mystic", "Priest of Titania", "Timberwatch Elf",
              "Nyxborn Hydra", "Lead the Stampede", "Fyndhorn Elves", "Wellwisher",
              "Llanowar Elves", "Quirion Ranger", "Avenging Hunter", "Masked Vandal",
              "Generous Ent"},
    "Affinity": {"Myr Enforcer", "Thoughtcast", "Ichor Wellspring", "Krark-Clan Shaman",
                 "Refurbished Familiar", "Writhing Chrysalis", "Seat of the Synod",
                 "Black Mage's Rod", "Hunter's Blowgun", "Cryogen Relic",
                 "Makeshift Munitions", "Blood Fountain", "Vault of Whispers"},
    "Wildfire": {"Cleansing Wildfire", "Fanatical Offering", "Toxin Analysis",
                 "Eviscerator's Insight", "Cast Down", "Pulse of Murasa",
                 "Twisted Landscape"},
}


def detect_archetype(hand_text: str) -> str:
    cards = [c.strip() for c in hand_text.split(";")]
    card_set = set(cards)
    best = None
    best_score = 0
    for arch, sigs in ARCHETYPE_SIGS.items():
        score = len(card_set & sigs)
        if score > best_score:
            best_score = score
            best = arch
    return best or "Unknown"


def read_game(path: Path) -> tuple[str, str, str, str]:
    """Return (agent_deck, opp_deck, winner, trigger_ep) for an eval game log.

    Prefers the authoritative ``Reason: Eval ep=X vs EVAL-CP7(agent=A, opp=B)``
    line over card-based archetype heuristics.
    """
    agent = "Unknown"
    opp = "Unknown"
    winner = "Unknown"
    trigger_ep = ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("Winner:"):
                    winner = line.split(":", 1)[1].strip()
                elif line.startswith("Reason:") and "EVAL-CP7" in line:
                    m_ag = re.search(r"agent=Deck - ([^,)]+)", line)
                    m_opp = re.search(r"opp=Deck - ([^)]+)", line)
                    m_ep = re.search(r"ep=(\d+)", line)
                    if m_ag:
                        agent = m_ag.group(1).strip()
                    if m_opp:
                        opp = m_opp.group(1).strip()
                    if m_ep:
                        trigger_ep = m_ep.group(1)
                elif agent == "Unknown" and "MULLIGAN_DECISION: player=EvalRL" in line:
                    m = re.search(r"hand=\[([^\]]+)\]", line)
                    if m:
                        agent = detect_archetype(m.group(1))
    except OSError:
        pass
    return agent, opp, winner, trigger_ep


def main() -> int:
    if not EVAL_DIR.exists():
        print(f"No eval dir at {EVAL_DIR}")
        return 1
    per_agent = defaultdict(lambda: [0, 0])  # agent deck -> [wins, total]
    per_cell = defaultdict(lambda: [0, 0])   # (agent, opp) -> [wins, total]
    per_ep = defaultdict(lambda: [0, 0])
    total = [0, 0]
    for game_path in sorted(EVAL_DIR.glob("*.txt")):
        agent, opp, winner, trigger_ep = read_game(game_path)
        if winner == "Unknown":
            continue
        is_win = int(winner.startswith("EvalRL"))
        per_agent[agent][0] += is_win
        per_agent[agent][1] += 1
        per_cell[(agent, opp)][0] += is_win
        per_cell[(agent, opp)][1] += 1
        if trigger_ep:
            per_ep[trigger_ep][0] += is_win
            per_ep[trigger_ep][1] += 1
        total[0] += is_win
        total[1] += 1

    src = "ARCHIVE (cumulative)" if EVAL_DIR == ARCHIVE_DIR else "LIVE (sliding 50-file window)"
    print(f"=== Eval winrate by agent deck — source={src} ===")
    print(f"    path: {EVAL_DIR}\n")
    print(f"  {'agent deck':>25} {'wins':>6} {'played':>7} {'winrate':>8}")
    for arch in sorted(per_agent.keys()):
        w, t = per_agent[arch]
        wr = w / t if t else 0
        print(f"  {arch:>25} {w:>6} {t:>7} {wr:>8.2%}")
    w, t = total
    print(f"  {'OVERALL':>25} {w:>6} {t:>7} {(w/t if t else 0):>8.2%}")

    if per_cell:
        print("\n=== Per-cell (agent vs opp) ===\n")
        print(f"  {'agent':>25} {'opp':>25} {'wins':>6} {'played':>7} {'winrate':>8}")
        for (ag, op), (w, t) in sorted(per_cell.items()):
            wr = w / t if t else 0
            print(f"  {ag:>25} {op:>25} {w:>6} {t:>7} {wr:>8.2%}")

    if per_ep:
        print("\n=== By trigger episode ===\n")
        for ep in sorted(per_ep.keys(), key=lambda e: int(e)):
            w, t = per_ep[ep]
            wr = w / t if t else 0
            print(f"  ep={ep}  {w}/{t} = {wr:.2%}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
