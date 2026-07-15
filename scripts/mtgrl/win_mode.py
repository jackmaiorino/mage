#!/usr/bin/env python3
"""Win-mode breakdown: of the agent's WINS, how many are COMBO kills vs BEATDOWN?

Disambiguates "better deck raised winrate" into combo-reach vs creature-beatdown
(the consistency deck has a big creature base + Mesmeric Fiend, so it can win without
the Spy combo). Needed to interpret the brewing ablation: a winrate lift via beatdown
is NOT evidence the mana tuning fixed combo-reach.

Per game:
  COMBO win  = win AND agent cast Balustrade Spy AND full self-mill (graveyard ballooned >=25)
               (the Spy->mill->Dread Return/Lotleth line)
  BEATDOWN win = win AND NOT combo-executed (won via creature damage / other)

Also reports combo-execution rate and how opponent died (life trajectory) when parseable.

Usage: python win_mode.py DIR [DIR ...]
"""
import glob, json, os, re, sys

STATE_RE = re.compile(r"PlayerRL1 L(-?\d+) H(\d+).*?G(\d+).*?\|\|\s*\S+ L(-?\d+)")


def parse_game(path):
    result = None; cast = False; max_gy = 0; opp_min_life = 99
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("RESULT:"):
                r = s.split(":", 1)[1].strip().upper()
                result = "win" if r.startswith("WIN") else ("loss" if r.startswith("LOSS") else None)
            elif "STATE:" in s:
                m = STATE_RE.search(s)
                if m:
                    max_gy = max(max_gy, int(m.group(3)))
                    opp_min_life = min(opp_min_life, int(m.group(4)))
            elif "REPLAY_DECISION_JSON:" in s:
                try:
                    obj = json.loads(s.split("REPLAY_DECISION_JSON:", 1)[1].strip())
                except Exception:
                    continue
                if obj.get("player") != "PlayerRL1":
                    continue
                ct = obj.get("chosen_texts") or []
                if any("Balustrade Spy" in t and t.lower().startswith("cast") for t in ct):
                    cast = True
    if result is None:
        return None
    executed = cast and max_gy >= 25
    return dict(result=result, cast=cast, executed=executed,
                opp_min_life=(opp_min_life if opp_min_life < 99 else None))


def main():
    dirs = sys.argv[1:]
    logs = []
    for d in dirs:
        logs += glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    games = [g for g in (parse_game(p) for p in logs) if g]
    n = len(games)
    wins = [g for g in games if g["result"] == "win"]
    combo_wins = [g for g in wins if g["executed"]]
    beat_wins = [g for g in wins if not g["executed"]]
    print(f"games={n}  WR={len(wins)/n:.3f}  ({len(wins)} wins)")
    print(f"  COMBO wins (cast Spy + full mill gy>=25): {len(combo_wins)} = {len(combo_wins)/max(1,len(wins)):.0%} of wins, {len(combo_wins)/n:.1%} of games")
    print(f"  BEATDOWN/other wins (no combo execution):  {len(beat_wins)} = {len(beat_wins)/max(1,len(wins)):.0%} of wins, {len(beat_wins)/n:.1%} of games")
    cast = [g for g in games if g["cast"]]
    execd = [g for g in games if g["executed"]]
    print(f"  cast-Spy rate={len(cast)/n:.0%}  combo-execution rate (of all games)={len(execd)/n:.0%}")
    print("\nREAD: if a winrate lift is mostly BEATDOWN wins, the deck improved a creature plan B, "
          "NOT combo-reach -- interpret the brewing ablation accordingly.")


if __name__ == "__main__":
    main()
