#!/usr/bin/env python3
"""Diagnose the castable-Balustrade-Spy gap: WM01 reaches a castable state in
~56% of games vs baseline ~75%, while willingness-to-cast is ~100% for both.
This decomposes WHY, per game, from REPLAY_DECISION_JSON (PlayerRL1) logs.

Signals (exact, from logs):
  reached_castable  = "Cast Balustrade Spy" ever in candidate_texts
  cast_spy          = "Cast Balustrade Spy" ever selected_text
  targeted_self     = "Balustrade Spy (you)" ever selected (the combo line)
  spy_in_hand_ever  = "Balustrade Spy" ever in hand
Failure classification (when not reached_castable):
  no_spy_in_hand    -> never drew/found Spy (draw/dig variance)
  spy_stuck_no_mana -> held Spy but never assembled {2}{U}
  ended_early       -> neither (game resolved before, e.g. died/decked)

Spy deck mana facts: only 4 lands (3 Forest, 1 Swamp); Balustrade Spy = {2}{U};
the ONLY blue sources are any-color producers Saruli Caretaker + Lotus Petal.

Usage: python castable_spy_diagnostic.py --wm DIR... --baseline DIR... [--samples 6]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

RL = "PlayerRL1"
LANDS = {"Forest", "Swamp"}
BLUE_SRC = {"Saruli Caretaker", "Lotus Petal"}  # only any-color producers = only U
OTHER_MANA = {"Overgrown Battlement", "Wall of Roots", "Elves of Deep Shadow",
              "Tinder Wall", "Quirion Ranger"}
FETCH = {"Land Grant", "Gatecreeper Vine", "Generous Ent", "Sagu Wildling"}
DIG = {"Lead the Stampede", "Winding Way"}
MULL_ACTIONS = {"MULLIGAN", "LONDON_MULLIGAN"}


def iter_decisions(path):
    outcome = None
    decs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("RESULT:"):
                r = s.split(":", 1)[1].strip().upper()
                outcome = "win" if r.startswith("WIN") else ("loss" if r.startswith("LOSS") else None)
            elif "REPLAY_DECISION_JSON:" in s:
                try:
                    o = json.loads(s.split("REPLAY_DECISION_JSON:", 1)[1].strip())
                except Exception:
                    continue
                if o.get("player") != RL:
                    continue
                decs.append(o)
    return outcome, decs


def card_played(chosen_texts, cardset):
    """Count distinct cards from cardset that were Cast/Played in this decision."""
    hits = set()
    for t in chosen_texts or []:
        if t.startswith("Cast ") or t.startswith("Play "):
            name = t.split(" ", 1)[1].strip()
            if name in cardset:
                hits.add(name)
    return hits


def analyze_game(outcome, decs):
    g = {
        "outcome": outcome, "n_dec": len(decs),
        "reached_castable": False, "turn_castable": None,
        "cast_spy": False, "targeted_self": False,
        "spy_in_hand_ever": False, "turn_spy_in_hand": None,
        "mulligans": 0, "keep_size": None,
        "keep_lands": 0, "keep_blue": 0, "keep_dig": 0, "keep_spy": 0, "keep_mana_any": 0,
        "played_blue_src": set(), "played_lands": 0, "played_other_mana": set(),
        "played_dig": 0, "max_turn": 0, "final_lib": None,
        "failure": None,
    }
    played_lands = 0
    first_nonmull_hand = None
    for o in decs:
        at = o.get("action_type", "")
        turn = int(o.get("turn", 0) or 0)
        g["max_turn"] = max(g["max_turn"], turn)
        g["mulligans"] = max(g["mulligans"], int(o.get("mulligans_taken", 0) or 0))
        hand = o.get("hand", []) or []
        cand = o.get("candidate_texts", []) or []
        sel = o.get("selected_text", "") or ""
        if o.get("library_size") is not None:
            g["final_lib"] = int(o["library_size"])
        # castable / cast / target
        if "Cast Balustrade Spy" in cand:
            if not g["reached_castable"]:
                g["reached_castable"] = True
                g["turn_castable"] = turn
        if sel == "Cast Balustrade Spy":
            g["cast_spy"] = True
        if sel == "Balustrade Spy (you)":
            g["targeted_self"] = True
        # spy in hand
        if "Balustrade Spy" in hand:
            g["spy_in_hand_ever"] = True
            if g["turn_spy_in_hand"] is None:
                g["turn_spy_in_hand"] = turn
        # first non-mulligan hand = opening keep
        if at not in MULL_ACTIONS and first_nonmull_hand is None:
            first_nonmull_hand = list(hand)
        # mana development (cast/played sources)
        g["played_blue_src"] |= card_played(o.get("chosen_texts"), BLUE_SRC)
        g["played_other_mana"] |= card_played(o.get("chosen_texts"), OTHER_MANA)
        for t in o.get("chosen_texts", []) or []:
            if t.startswith("Play ") and t.split(" ", 1)[1].strip() in LANDS:
                played_lands += 1
        g["played_dig"] += len(card_played(o.get("chosen_texts"), DIG))
    g["played_lands"] = played_lands
    g["played_blue_src"] = sorted(g["played_blue_src"])
    g["played_other_mana"] = sorted(g["played_other_mana"])
    # opening keep profile
    if first_nonmull_hand is not None:
        g["keep_size"] = len(first_nonmull_hand)
        g["keep_lands"] = sum(1 for c in first_nonmull_hand if c in LANDS)
        g["keep_blue"] = sum(1 for c in first_nonmull_hand if c in BLUE_SRC)
        g["keep_dig"] = sum(1 for c in first_nonmull_hand if c in DIG)
        g["keep_spy"] = sum(1 for c in first_nonmull_hand if c == "Balustrade Spy")
        g["keep_mana_any"] = sum(1 for c in first_nonmull_hand
                                 if c in LANDS or c in BLUE_SRC or c in OTHER_MANA or c in FETCH)
    # failure classification
    if not g["reached_castable"]:
        if not g["spy_in_hand_ever"]:
            g["failure"] = "no_spy_in_hand"
        elif not g["played_blue_src"]:
            g["failure"] = "spy_stuck_no_blue"
        else:
            g["failure"] = "spy_stuck_other"
    return g


def load(dirs):
    games = []
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "*.txt"), recursive=True):
            outcome, decs = iter_decisions(p)
            if outcome in ("win", "loss") and len(decs) >= 3:
                gg = analyze_game(outcome, decs)
                gg["_path"] = p
                games.append(gg)
    return games


def summarize(name, games):
    n = len(games)
    if not n:
        print(f"{name}: no games"); return
    def rate(pred):
        m = [g for g in games if pred(g)]
        return len(m), 100.0 * len(m) / n
    reached = [g for g in games if g["reached_castable"]]
    cast = [g for g in games if g["cast_spy"]]
    print(f"\n===== {name}: {n} games =====")
    print(f"  winrate              : {rate(lambda g: g['outcome']=='win')[1]:.1f}%")
    print(f"  reached castable Spy : {rate(lambda g: g['reached_castable'])[1]:.1f}%  ({len(reached)}/{n})")
    print(f"  cast Spy             : {rate(lambda g: g['cast_spy'])[1]:.1f}%")
    print(f"  willingness (cast|reached): {100.0*len(cast)/len(reached) if reached else float('nan'):.1f}%")
    print(f"  targeted self (combo): {rate(lambda g: g['targeted_self'])[1]:.1f}%")
    print(f"  Spy in hand ever     : {rate(lambda g: g['spy_in_hand_ever'])[1]:.1f}%")
    # failure breakdown (of NOT reached)
    notr = [g for g in games if not g["reached_castable"]]
    fc = defaultdict(int)
    for g in notr:
        fc[g["failure"]] += 1
    print(f"  NOT-reached ({len(notr)}): " + ", ".join(f"{k}={v} ({100.0*v/n:.0f}% of all)" for k, v in sorted(fc.items())))
    # mulligan + keep profile
    import statistics as st
    mulls = [g["mulligans"] for g in games]
    keepl = [g["keep_lands"] for g in games if g["keep_size"] is not None]
    keepb = [g["keep_blue"] for g in games if g["keep_size"] is not None]
    keepm = [g["keep_mana_any"] for g in games if g["keep_size"] is not None]
    keeps = [g["keep_size"] for g in games if g["keep_size"] is not None]
    keepspy = [g["keep_spy"] for g in games if g["keep_size"] is not None]
    print(f"  mulligans: mean={st.mean(mulls):.2f} median={st.median(mulls)}  | keep_size mean={st.mean(keeps):.2f}")
    print(f"  opening keep: lands mean={st.mean(keepl):.2f} | blue-src mean={st.mean(keepb):.2f} | "
          f"any-mana mean={st.mean(keepm):.2f} | has-Spy {100.0*sum(1 for x in keepspy if x)/len(keepspy):.0f}%")
    print(f"  turn first castable (reached games): mean={st.mean([g['turn_castable'] for g in reached]):.1f}" if reached else "  (none reached)")
    print(f"  played blue source (Saruli/Lotus): {rate(lambda g: bool(g['played_blue_src']))[1]:.1f}% | "
          f"played dig spell: {rate(lambda g: g['played_dig']>0)[1]:.1f}%")
    return fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=0, help="dump N not-reached sample paths per model")
    args = ap.parse_args()
    wm = load(args.wm)
    base = load(args.baseline)
    summarize("WM01", wm)
    summarize("BASELINE", base)
    if args.samples:
        for name, games in [("WM01", wm), ("BASELINE", base)]:
            notr = [g for g in games if not g["reached_castable"]]
            print(f"\n--- {name} not-reached samples (path | failure | mulls | keep_lands/blue | spy_in_hand) ---")
            for g in notr[:args.samples]:
                print(f"  {os.path.basename(g['_path'])} | {g['failure']} | mull={g['mulligans']} | "
                      f"keepL/B={g['keep_lands']}/{g['keep_blue']} | spy_hand={g['spy_in_hand_ever']} | "
                      f"playedBlue={g['played_blue_src']} | maxturn={g['max_turn']}")


if __name__ == "__main__":
    main()
