"""Disambiguate play-skill vs deck-speed for the 4-mana-tempo bottleneck.

For never-cast-Spy LOSS games, measure how the game went:
  - max_turn: how long the game lasted (game length)
  - max_mana_sources: peak count of the agent's mana-producing permanents in play
    (lands + dorks: Forest/Swamp/Wall of Roots/Overgrown Battlement/Saruli
    Caretaker/Elves of Deep Shadow/Lotus Petal/Tinder Wall)
Balustrade Spy = {3}{B} (needs 4 mana). So:
  - long game (turn>=8) but max_mana_sources < 4  -> had TIME, never ramped to 4
    => durdle / play-sequencing problem (deck HAS the ramp; not deploying it).
  - long game AND reached >=4 sources but still never cast -> had mana+time,
    Spy not deployed (timing/simultaneity) => play problem.
  - short game (turn<=6) and < 4 sources -> Affinity's clock killed it first
    => deck/matchup speed problem.

Usage: python never_cast_tempo.py <run_game_logs_dir>
"""
import json
import os
import re
import sys

DECJSON = re.compile(r'REPLAY_DECISION_JSON:\s*(\{.*\})\s*$', re.MULTILINE)
STATE = re.compile(r'PlayerRL1 L-?\d+ H\d+\[[^\]]*\] B\d+\[([^\]]*)\]')
PRODUCERS = ("Forest", "Swamp", "Wall of Roots", "Overgrown Battlement",
             "Saruli Caretaker", "Elves of Deep Shadow", "Lotus Petal", "Tinder Wall")


def count_producers(bf: str) -> int:
    return sum(bf.count(p) for p in PRODUCERS)


def analyze(txt):
    cast = '"selected_text":"Cast Balustrade Spy' in txt or \
        '"chosen_texts":["Cast Balustrade Spy' in txt
    drew = False
    max_turn = 0
    for m in DECJSON.finditer(txt):
        try:
            j = json.loads(m.group(1))
        except Exception:
            continue
        if "Balustrade Spy" in " ".join(j.get("hand", []) or []):
            drew = True
        try:
            max_turn = max(max_turn, int(j.get("turn", 0)))
        except Exception:
            pass
    max_src = 0
    for m in STATE.finditer(txt):
        max_src = max(max_src, count_producers(m.group(1)))
    return cast, drew, max_turn, max_src


def main():
    d = sys.argv[1]
    rows = []
    for f in os.listdir(d):
        if not f.endswith(".txt"):
            continue
        with open(os.path.join(d, f), encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        cast, drew, mt, ms = analyze(txt)
        loss = "RESULT: WIN" not in txt
        rows.append((cast, drew, loss, mt, ms, os.path.basename(f)))

    nc_loss = [r for r in rows if (not r[0]) and r[2]]
    print(f"games={len(rows)} never-cast-Spy losses={len(nc_loss)}")
    print(f"{'maxTurn':>7} {'maxManaSrc':>10} {'drewSpy':>7}  file")
    for r in sorted(nc_loss, key=lambda x: x[3]):
        print(f"{r[3]:>7} {r[4]:>10} {str(r[1]):>7}  {r[5]}")

    long_no4 = [r for r in nc_loss if r[3] >= 8 and r[4] < 4]
    long_4 = [r for r in nc_loss if r[3] >= 8 and r[4] >= 4]
    short = [r for r in nc_loss if r[3] <= 6]
    mid = [r for r in nc_loss if 6 < r[3] < 8]
    print(f"\nDURDLE/play  (turn>=8, <4 sources)      : {len(long_no4)}")
    print(f"PLAY-timing  (turn>=8, reached >=4 src)  : {len(long_4)}")
    print(f"mid          (turn 7)                    : {len(mid)}")
    print(f"CLOCK/deck   (turn<=6)                   : {len(short)}")
    if nc_loss:
        avg_t = sum(r[3] for r in nc_loss) / len(nc_loss)
        avg_s = sum(r[4] for r in nc_loss) / len(nc_loss)
        print(f"\navg max_turn={avg_t:.1f}  avg max_mana_sources={avg_s:.1f}")
        print("=> durdle/play buckets dominant + long games => PLAY-skill (ramp/sequencing). "
              "short/clock dominant => deck-speed vs Affinity.")


if __name__ == "__main__":
    main()
