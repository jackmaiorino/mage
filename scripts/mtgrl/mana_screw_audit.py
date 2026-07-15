#!/usr/bin/env python3
"""Mana-sequencing sliver audit (Codex's last rigor gap before brewing).

The engine already proved Spy was never CASTABLE in the 19 mana-screw games (never
offered). The only remaining question: could DIFFERENT mana sequencing / deployment
have reached {3}{B}? Two cheap log-decisive signals per game:

  (1) BLACK availability: did the agent EVER have any black source (Swamp, Elves of
      Deep Shadow, Saruli Caretaker [any color], Lotus Petal [any color]) on board OR
      in hand? If NEVER -> {B} was impossible regardless of sequencing = genuine
      black-screw (deck variance, not policy).
  (2) SITTING on deployable mana: did the agent hold an unconditionally-deployable
      source (a LAND = 1/turn, or LOTUS PETAL = {0}) in hand across >=2 Spi-in-hand
      decisions while the board stayed short -> a sequencing miss (could have added
      mana and didn't).

Classifies the 19: BLACK-SCREWED (no black ever) / DEVELOPED-ALL (no sitting, just
short) / POSSIBLE-SEQUENCING-MISS (sat on a land/Petal). Many in the last bucket =>
narrow sequencing bug; otherwise => deck/variance, brew.

Usage: python mana_screw_audit.py DIR [DIR ...]
"""
import glob
import json
import os
import re
import sys

# STATE: PlayerRL1 L# H#[..] B#[..] G#[..] X#  (B/G bracket only present when >0)
STATE_RE = re.compile(
    r"PlayerRL1 L(-?\d+) H(\d+)(?:\[(.*?)\])? B(\d+)(?:\[(.*?)\])? G(\d+)(?:\[(.*?)\])?")
TURN_RE = re.compile(r"DECISION #\d+ - Turn (\d+)")

BLACK_SOURCES = {"Swamp", "Elves of Deep Shadow", "Saruli Caretaker", "Lotus Petal"}
DEPLOYABLE_NOW = {"Swamp", "Forest", "Lotus Petal"}  # land (1/turn) or {0} rock


def names(bracket):
    if not bracket:
        return []
    return [c.split(",")[0].strip() for c in bracket.split(";") if c.strip()]


def parse_game(path):
    cur_turn = 0
    spy_cast = False
    spy_offered = False
    spy_in_hand_turns = set()
    black_ever = False
    # track held deployable sources across spy-in-hand decisions
    held_deployable_runs = 0   # decisions where a land/Petal sat in hand while spy in hand
    board_sources_peak = 0
    result = None
    # turn-based missed-land-drop tracking: per turn, (land_in_hand?, board_land_count)
    turn_landhand = {}
    turn_boardlands = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            mt = TURN_RE.search(s)
            if mt:
                cur_turn = int(mt.group(1))
            if s.startswith("RESULT:"):
                r = s.split(":", 1)[1].strip().upper()
                result = "win" if r.startswith("WIN") else ("loss" if r.startswith("LOSS") else None)
            if "STATE:" in s:
                m = STATE_RE.search(s)
                if m:
                    hand = names(m.group(3)); board = names(m.group(5))
                    if "Balustrade Spy" in hand:
                        spy_in_hand_turns.add(cur_turn)
                        # black anywhere?
                        if any(c in BLACK_SOURCES for c in hand + board):
                            black_ever = True
                        # sitting on a deployable mana source while spy stuck?
                        if any(c in DEPLOYABLE_NOW for c in hand):
                            held_deployable_runs += 1
                    # board mana-source count (rough)
                    bs = sum(1 for c in board if c in (
                        "Swamp", "Forest", "Overgrown Battlement", "Saruli Caretaker",
                        "Wall of Roots", "Elves of Deep Shadow", "Lotus Petal", "Tinder Wall"))
                    board_sources_peak = max(board_sources_peak, bs)
                    # per-turn land-in-hand + board-land tracking (for missed-land-drop)
                    bl = sum(1 for c in board if c in ("Swamp", "Forest"))
                    lih = any(c in ("Swamp", "Forest") for c in hand)
                    turn_landhand[cur_turn] = turn_landhand.get(cur_turn, False) or lih
                    turn_boardlands[cur_turn] = max(turn_boardlands.get(cur_turn, 0), bl)
            elif "REPLAY_DECISION_JSON:" in s:
                try:
                    obj = json.loads(s.split("REPLAY_DECISION_JSON:", 1)[1].strip())
                except Exception:
                    continue
                if obj.get("player") != "PlayerRL1":
                    continue
                cands = obj.get("candidate_texts") or []
                chosen = obj.get("chosen_texts") or []
                if any("Balustrade Spy" in t and t.lower().startswith("cast") for t in cands):
                    spy_offered = True
                if any("Balustrade Spy" in t and t.lower().startswith("cast") for t in chosen):
                    spy_cast = True
    # missed land drops = turns where a land was in hand but board lands did not increase next turn
    missed_drops = 0
    turns = sorted(turn_landhand)
    for i, t in enumerate(turns[:-1]):
        nxt = turns[i + 1]
        if turn_landhand[t] and turn_boardlands.get(nxt, 0) <= turn_boardlands.get(t, 0):
            missed_drops += 1
    return dict(result=result, spy_cast=spy_cast, spy_offered=spy_offered,
                spy_in_hand=len(spy_in_hand_turns) > 0, black_ever=black_ever,
                held_runs=held_deployable_runs, board_peak=board_sources_peak,
                missed_drops=missed_drops, path=path)


def main():
    dirs = sys.argv[1:]
    logs = []
    for d in dirs:
        logs += glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    games = [parse_game(p) for p in logs]
    games = [g for g in games if g["result"] is not None]
    # mana-screw = never-cast LOSS, never offered, but Spy was in hand
    screw = [g for g in games if (not g["spy_cast"]) and g["result"] == "loss"
             and (not g["spy_offered"]) and g["spy_in_hand"]]
    print(f"games={len(games)}  mana-screw (never-cast loss, Spy in hand, never castable) = {len(screw)}\n")
    if not screw:
        print("none"); return

    black_screw = [g for g in screw if not g["black_ever"]]
    has_black = [g for g in screw if g["black_ever"]]
    sitting = [g for g in has_black if g["held_runs"] >= 2]
    developed = [g for g in has_black if g["held_runs"] < 2]

    print("=== mana-screw breakdown ===")
    print(f"  BLACK-SCREWED (no black source EVER on board/hand): {len(black_screw)} "
          f"({len(black_screw)/len(screw):.0%}) -> {{B}} impossible, pure deck variance")
    print(f"  had black, POSSIBLE-SEQUENCING-MISS (sat on land/Lotus Petal >=2 decisions): {len(sitting)} "
          f"({len(sitting)/len(screw):.0%})")
    print(f"  had black, DEVELOPED-ALL (no sitting, just short of 4 mana): {len(developed)} "
          f"({len(developed)/len(screw):.0%})")
    import statistics as st
    print(f"\n  board mana-source peak (all screw): med={st.median([g['board_peak'] for g in screw]):.0f}")
    # Turn-based missed-land-drop = the REAL sequencing-miss signal (vs per-decision artifact)
    real_miss = [g for g in screw if g["missed_drops"] >= 1]
    print(f"\n  TURN-BASED missed land drops (land in hand, board lands didn't grow next turn):")
    print(f"    games with >=1 missed land drop: {len(real_miss)} ({len(real_miss)/len(screw):.0%})")
    print(f"    total missed drops across screw games: {sum(g['missed_drops'] for g in screw)}")
    print("\n  flagged (per-decision 'sitting') game paths:")
    for g in sitting:
        print(f"    held_runs={g['held_runs']} missed_drops={g['missed_drops']} board_peak={g['board_peak']} {os.path.basename(g['path'])}")
    print("\nREAD: BLACK-SCREWED + DEVELOPED-ALL = deck/variance (brew). "
          "REAL signal = turn-based missed land drops; near-0 => the 'sitting' flag was per-decision artifact => deck/variance.")


if __name__ == "__main__":
    main()
