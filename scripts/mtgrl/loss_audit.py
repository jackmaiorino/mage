#!/usr/bin/env python3
"""Conditional loss audit for the Spy combo (Codex-prescribed before any further run).

Proves the residual failure mode rather than inferring it from NO_BOARD/executed.
Splits win-rate by: Spy-cast vs not, executed-mill vs fizzle, and characterizes
LOSSES by game length + opponent clock to estimate RECOVERABLE (durdle/policy) vs
UNWINNABLE (rushed/matchup-ceiling) -- the distinction that gates whether a
trajectory-context experiment is worth it vs the strategic fork.

Parses game logs:
  - "DECISION #N - Turn T ..." headers -> turn numbers
  - "STATE: ... PlayerRL1 L<life> H<n>[..] B<n>[..] G<n>[..] X<n> || EvalBot.. L<life> H<n> B<n>[..] .."
  - REPLAY_DECISION_JSON chosen_texts -> "Cast Balustrade Spy"
  - "RESULT: WIN|LOSS"

Usage: python loss_audit.py DIR [DIR ...]
"""
import glob
import json
import os
import re
import sys

STATE_RE = re.compile(
    r"PlayerRL1 L(-?\d+) H(\d+).*?B(\d+).*?G(\d+).*?\|\|\s*\S+ L(-?\d+) H(\d+) B(\d+)")
TURN_RE = re.compile(r"DECISION #\d+ - Turn (\d+)")


def parse_game(path):
    result = None
    cast_spy = False
    cast_turn = None
    last_turn = 0
    our_min_life = 99
    opp_max_board = 0
    our_max_gy = 0
    cur_turn = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            mt = TURN_RE.search(s)
            if mt:
                cur_turn = int(mt.group(1)); last_turn = max(last_turn, cur_turn)
            if s.startswith("RESULT:"):
                r = s.split(":", 1)[1].strip().upper()
                result = "win" if r.startswith("WIN") else ("loss" if r.startswith("LOSS") else None)
            elif "STATE:" in s:
                m = STATE_RE.search(s)
                if m:
                    our_life = int(m.group(1)); our_board = int(m.group(3)); our_gy = int(m.group(4))
                    opp_board = int(m.group(7))
                    our_min_life = min(our_min_life, our_life)
                    opp_max_board = max(opp_max_board, opp_board)
                    our_max_gy = max(our_max_gy, our_gy)
            elif "REPLAY_DECISION_JSON:" in s:
                try:
                    obj = json.loads(s.split("REPLAY_DECISION_JSON:", 1)[1].strip())
                except Exception:
                    continue
                if obj.get("player") != "PlayerRL1":
                    continue
                ct = obj.get("chosen_texts") or []
                if any("Balustrade Spy" in t and t.lower().startswith("cast") for t in ct):
                    cast_spy = True
                    if cast_turn is None:
                        cast_turn = cur_turn
    if result is None:
        return None
    # executed-mill proxy: graveyard ballooned (full self-mill ~ whole library to GY)
    executed = cast_spy and our_max_gy >= 25
    return dict(result=result, cast=cast_spy, cast_turn=cast_turn, last_turn=last_turn,
                min_life=(our_min_life if our_min_life < 99 else None),
                opp_max_board=opp_max_board, max_gy=our_max_gy, executed=executed)


def wr(games):
    return sum(1 for g in games if g["result"] == "win") / len(games) if games else float("nan")


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: loss_audit.py DIR [DIR ...]"); sys.exit(1)
    logs = []
    for d in dirs:
        logs += glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    games = [g for g in (parse_game(p) for p in logs) if g]
    n = len(games)
    print(f"games={n}  overall WR={wr(games):.3f}  dirs={[os.path.basename(d) for d in dirs]}\n")

    cast = [g for g in games if g["cast"]]
    nocast = [g for g in games if not g["cast"]]
    execd = [g for g in cast if g["executed"]]
    fizzle = [g for g in cast if not g["executed"]]
    print(f"cast Spy:        {len(cast):3d}/{n} ({len(cast)/n:.0%})  WR|cast    = {wr(cast):.3f}")
    print(f"  executed-mill: {len(execd):3d}      WR|executed = {wr(execd):.3f}")
    print(f"  fizzle (lands remained / partial): {len(fizzle):3d}  WR|fizzle = {wr(fizzle):.3f}")
    print(f"never cast Spy:  {len(nocast):3d}/{n} ({len(nocast)/n:.0%})  WR|nocast  = {wr(nocast):.3f}\n")

    losses = [g for g in games if g["result"] == "loss"]
    print(f"--- LOSS breakdown (n={len(losses)}) ---")
    l_cast = [g for g in losses if g["cast"]]
    l_nocast = [g for g in losses if not g["cast"]]
    print(f"losses where Spy WAS cast:   {len(l_cast):3d} ({len(l_cast)/max(1,len(losses)):.0%})")
    print(f"losses where Spy NEVER cast: {len(l_nocast):3d} ({len(l_nocast)/max(1,len(losses)):.0%})")

    def chars(gs, label):
        if not gs:
            print(f"  [{label}] none"); return
        import statistics as st
        lt = [g["last_turn"] for g in gs]
        ob = [g["opp_max_board"] for g in gs]
        ml = [g["min_life"] for g in gs if g["min_life"] is not None]
        print(f"  [{label}] n={len(gs)}  last_turn med={st.median(lt):.0f} (min {min(lt)}, max {max(lt)})  "
              f"opp_max_board med={st.median(ob):.0f}  our_min_life med={st.median(ml) if ml else '?'}")

    chars(l_nocast, "never-cast losses")
    chars(l_cast, "cast-but-lost")

    # Recoverability proxy for never-cast losses: long game (had time/mana, durdled) vs rushed.
    rec = [g for g in l_nocast if g["last_turn"] >= 7]
    rush = [g for g in l_nocast if g["last_turn"] <= 5]
    mid = [g for g in l_nocast if 5 < g["last_turn"] < 7]
    print(f"\n  never-cast loss split: durdle/recoverable (turn>=7)={len(rec)}  "
          f"mid(6)={len(mid)}  rushed/ceiling (turn<=5)={len(rush)}")
    print("  (recoverable = had time but never comboed = policy/tempo failure; "
          "rushed = killed before a combo window = matchup ceiling)")

    # turn-of-cast distribution for wins
    win_cast = [g for g in cast if g["result"] == "win" and g["cast_turn"]]
    if win_cast:
        import statistics as st
        ct = [g["cast_turn"] for g in win_cast]
        print(f"\n  WINS via cast: cast_turn med={st.median(ct):.0f} (min {min(ct)}, max {max(ct)})")


if __name__ == "__main__":
    main()
