"""Why does the agent not cast Balustrade Spy in never-cast games?

For games where Cast Balustrade Spy was never chosen, parse REPLAY_DECISION_JSON
to determine the cause:
  - drew_spy: Balustrade Spy ever appeared in the agent's hand
  - had_blue: a blue source (Saruli Caretaker / Lotus Petal) ever on battlefield
    or in hand (Spy costs {2}{U}; these are the deck's only blue producers)
Classifies each never-cast game as:
  - mana_assembly: drew Spy but never had blue -> couldn't make {U} (deck/draw)
  - drew_spy_had_blue: drew Spy AND had blue but still never cast -> a real
    play/timing failure (the interesting bucket)
  - never_drew_spy: variance (deck consistency)
This separates a DECK-consistency ceiling from a PLAY-skill ceiling.

Usage: python never_cast_why.py <run_game_logs_dir>
"""
import json
import os
import re
import sys

DEC = re.compile(r'REPLAY_DECISION_JSON:\s*(\{.*\})\s*$', re.MULTILINE)
# Balustrade Spy costs {3}{B} (4 mana, one BLACK). Deck's black sources:
BLACK = ("Swamp", "Elves of Deep Shadow", "Saruli Caretaker", "Lotus Petal")


def analyze(txt):
    cast = '"selected_text":"Cast Balustrade Spy' in txt or \
        '"chosen_texts":["Cast Balustrade Spy' in txt
    drew_spy = False
    had_black = False
    max_mana = 0  # max battlefield permanents that could tap for mana, rough proxy for reaching 4
    for m in DEC.finditer(txt):
        try:
            j = json.loads(m.group(1))
        except Exception:
            continue
        hand = " ".join(j.get("hand", []) or [])
        if "Balustrade Spy" in hand:
            drew_spy = True
        inplay = hand + " " + str(j.get("visible_self_battlefield_names", ""))
        if any(b in inplay for b in BLACK):
            had_black = True
    return cast, drew_spy, had_black


def main():
    d = sys.argv[1]
    cats = {"mana_assembly": 0, "drew_spy_had_black": 0, "never_drew_spy": 0}
    never_cast = 0
    never_cast_loss = 0
    total = 0
    for f in os.listdir(d):
        if not f.endswith(".txt"):
            continue
        with open(os.path.join(d, f), encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        total += 1
        cast, drew, black = analyze(txt)
        if cast:
            continue
        never_cast += 1
        if "RESULT: WIN" not in txt:
            never_cast_loss += 1
        if not drew:
            cats["never_drew_spy"] += 1
        elif drew and not black:
            cats["mana_assembly"] += 1
        else:
            cats["drew_spy_had_black"] += 1
    print(f"games={total} never_cast_Spy={never_cast} (losses={never_cast_loss})")
    print("WHY never cast Spy:")
    for k, v in cats.items():
        print(f"  {k:20s}: {v}")
    print("\nSpy = {3}{B} (4 mana incl BLACK). mana_assembly = drew Spy but never had a "
          "black source in play (Swamp/Elves/Saruli/Lotus Petal). never_drew_spy = variance. "
          "drew_spy_had_black = had Spy + a black source but never reached castable 4-mana "
          "(tempo/sequencing — the bucket that could be play-skill OR deck-speed).")


if __name__ == "__main__":
    main()
