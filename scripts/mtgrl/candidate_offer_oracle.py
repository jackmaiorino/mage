#!/usr/bin/env python3
"""Resource oracle (Codex gate 2) for the never-cast losses.

The engine's legal candidate set is ground truth for "is Balustrade Spy castable
RIGHT NOW" (i.e. agent has >=4 mana incl. black AND Spy in hand). So for each game
where the agent never cast Spy:

  - Was "Cast Balustrade Spy" EVER offered as a legal candidate?
      offered + declined  -> RECOVERABLE (the resource line existed; policy passed it)
      never offered       -> REACH-STARVED (mana/card never came together)
  - For never-offered: was Spy ever even in hand (draw) vs in hand but never castable
    (mana screw) vs never drawn.
  - For offered-declined: board-creature count at the offer (>=3 => a finish was set up
    too => declining was a real winning-line miss; <3 => decline may be correct).

This is the resource upper-bound that proves/refutes "recoverable" before deciding
trajectory-train vs strategic fork.

Usage: python candidate_offer_oracle.py DIR [DIR ...]
"""
import glob
import json
import os
import re
import sys

STATE_RE = re.compile(r"PlayerRL1 L(-?\d+) H(\d+)\[(.*?)\] B(\d+)")
TURN_RE = re.compile(r"DECISION #\d+ - Turn (\d+)")


def parse_game(path):
    cur_turn = 0
    last_board = 0
    spy_cast = False
    spy_offered = 0          # # decisions where Cast Balustrade Spy was a legal candidate
    spy_offer_board = []     # board-creature count at each offer
    spy_in_hand_ever = False
    max_board = 0
    last_turn = 0
    result = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            mt = TURN_RE.search(s)
            if mt:
                cur_turn = int(mt.group(1)); last_turn = max(last_turn, cur_turn)
            if s.startswith("RESULT:"):
                r = s.split(":", 1)[1].strip().upper()
                result = "win" if r.startswith("WIN") else ("loss" if r.startswith("LOSS") else None)
            if "STATE:" in s:
                m = STATE_RE.search(s)
                if m:
                    last_board = int(m.group(4)); max_board = max(max_board, last_board)
                    hand_txt = m.group(3)
                    if "Balustrade Spy" in hand_txt:
                        spy_in_hand_ever = True
            elif "REPLAY_DECISION_JSON:" in s:
                try:
                    obj = json.loads(s.split("REPLAY_DECISION_JSON:", 1)[1].strip())
                except Exception:
                    continue
                if obj.get("player") != "PlayerRL1" or obj.get("action_type") == "MULLIGAN":
                    continue
                cands = obj.get("candidate_texts") or []
                offered = any("Balustrade Spy" in t and t.lower().startswith("cast") for t in cands)
                chosen = obj.get("chosen_texts") or []
                chose_spy = any("Balustrade Spy" in t and t.lower().startswith("cast") for t in chosen)
                if chose_spy:
                    spy_cast = True
                if offered:
                    spy_offered += 1
                    spy_offer_board.append(last_board)
    return dict(result=result, spy_cast=spy_cast, spy_offered=spy_offered,
                spy_offer_board=spy_offer_board, spy_in_hand_ever=spy_in_hand_ever,
                max_board=max_board, last_turn=last_turn)


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: candidate_offer_oracle.py DIR [DIR ...]"); sys.exit(1)
    logs = []
    for d in dirs:
        logs += glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    games = [parse_game(p) for p in logs]
    games = [g for g in games if g["result"] is not None]
    n = len(games)
    nocast = [g for g in games if not g["spy_cast"]]
    nocast_loss = [g for g in nocast if g["result"] == "loss"]
    print(f"games={n}  never-cast={len(nocast)}  never-cast LOSSES={len(nocast_loss)}\n")

    # Of never-cast LOSSES: offered-but-declined vs never-offered
    offered_declined = [g for g in nocast_loss if g["spy_offered"] > 0]
    never_offered = [g for g in nocast_loss if g["spy_offered"] == 0]
    print("=== never-cast LOSSES recoverability ===")
    print(f"  OFFERED but declined (RECOVERABLE - resource line existed): {len(offered_declined)} "
          f"({len(offered_declined)/max(1,len(nocast_loss)):.0%})")
    if offered_declined:
        import statistics as st
        offs = [g["spy_offered"] for g in offered_declined]
        # board at offers: was a finish set up too?
        ready = sum(1 for g in offered_declined if any(b >= 3 for b in g["spy_offer_board"]))
        print(f"    offers/game med={st.median(offs):.0f} (max {max(offs)}); "
              f"of these, {ready} had >=3 creatures at an offer (winning line available, declined)")
    print(f"  NEVER offered (REACH-STARVED - never castable): {len(never_offered)} "
          f"({len(never_offered)/max(1,len(nocast_loss)):.0%})")
    if never_offered:
        in_hand = sum(1 for g in never_offered if g["spy_in_hand_ever"])
        print(f"    of these: Spy in hand at some point (MANA-screw) = {in_hand}; "
              f"Spy never in hand (DRAW variance) = {len(never_offered) - in_hand}")
        import statistics as st
        mb = [g["max_board"] for g in never_offered]; lt = [g["last_turn"] for g in never_offered]
        print(f"    max_board med={st.median(mb):.0f}  last_turn med={st.median(lt):.0f}")

    print("\nREAD:")
    print("  high OFFERED-but-declined => RECOVERABLE policy failure (trajectory/search lever may help).")
    print("  high NEVER-offered + mana-screw/draw => REACH-STARVED = deck consistency => brewing/fork, not policy.")


if __name__ == "__main__":
    main()
