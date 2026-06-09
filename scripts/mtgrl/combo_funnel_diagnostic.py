#!/usr/bin/env python3
"""Full Spy-combo execution funnel + conditional win-rates, to find the biggest
execution leak (the pivot's target). Combo line: cast Balustrade Spy targeting
SELF (no lands in library) -> mill whole library -> Flashback Dread Return ->
return Lotleth Giant -> lethal Undergrowth.

Stages per game (from PlayerRL1 REPLAY_DECISION_JSON):
  spy_in_hand      "Balustrade Spy" ever in hand
  castable         "Cast Balustrade Spy" ever in candidate_texts
  cast_spy         "Cast Balustrade Spy" selected
  target_self      a Spy SELECT_TARGETS selecting "(you)"  (the combo line)
  target_opp       a Spy SELECT_TARGETS NOT "(you)"        (non-combo)
  milled_out       library_size hit 0 at some decision
  dread_return     "Dread Return" cast/flashback selected
  lotleth_return   Dread Return SELECT_TARGETS == Lotleth Giant (or LG enters after)
  win

Reports reach% and P(win | reached stage) for each stage, and the conditional
P(win) given target_self vs target_opp -- the key lever signal.
Usage: python combo_funnel_diagnostic.py --label NAME DIR [DIR ...]
"""
import argparse
import glob
import json
import os

RL = "PlayerRL1"


def parse(path):
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
                if o.get("player") == RL:
                    decs.append(o)
    return outcome, decs


def game_flags(outcome, decs):
    f = dict(spy_in_hand=False, castable=False, cast_spy=False, target_self=False,
             target_opp=False, milled_out=False, dread_return=False, lotleth_return=False,
             win=(outcome == "win"))
    for o in decs:
        at = o.get("action_type", "")
        sel = o.get("selected_text", "") or ""
        cand = o.get("candidate_texts", []) or []
        if "Balustrade Spy" in (o.get("hand", []) or []):
            f["spy_in_hand"] = True
        if "Cast Balustrade Spy" in cand:
            f["castable"] = True
        if sel == "Cast Balustrade Spy":
            f["cast_spy"] = True
        if at == "SELECT_TARGETS" and "Balustrade Spy" in sel:
            if "(you)" in sel:
                f["target_self"] = True
            else:
                f["target_opp"] = True
        if o.get("library_size") is not None and int(o["library_size"]) == 0:
            f["milled_out"] = True
        if ("Dread Return" in sel) and ("Cast" in sel or "Flashback" in sel):
            f["dread_return"] = True
        if at == "SELECT_TARGETS" and "Dread Return" in sel and "Lotleth Giant" in sel:
            f["lotleth_return"] = True
    return f


def load(dirs):
    out = []
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "*.txt"), recursive=True):
            outcome, decs = parse(p)
            if outcome in ("win", "loss") and len(decs) >= 3:
                out.append(game_flags(outcome, decs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="MODEL")
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    g = load(args.dirs)
    n = len(g)
    if not n:
        print("no games"); return
    nwin = sum(x["win"] for x in g)
    print(f"\n===== {args.label}: {n} games, winrate {100.0*nwin/n:.1f}% =====")
    stages = ["spy_in_hand", "castable", "cast_spy", "target_self", "milled_out",
              "dread_return", "lotleth_return"]
    print(f"{'stage':>15} {'reach%':>7} {'P(win|reach)':>13} {'n':>5}")
    prev = n
    for st in stages:
        sub = [x for x in g if x[st]]
        r = len(sub)
        pw = (100.0 * sum(x["win"] for x in sub) / r) if r else float("nan")
        drop = prev - r
        flag = "  <== biggest drop so far" if False else ""
        print(f"{st:>15} {100.0*r/n:>6.1f}% {pw:>12.1f}% {r:>5}   (drop {drop})")
        prev = r
    # key lever: target_self vs target_opp conditional winrate
    cast = [x for x in g if x["cast_spy"]]
    tself = [x for x in cast if x["target_self"]]
    topp = [x for x in cast if x["target_opp"] and not x["target_self"]]
    tnone = [x for x in cast if not x["target_self"] and not x["target_opp"]]
    def wr(s):
        return (100.0 * sum(x["win"] for x in s) / len(s)) if s else float("nan")
    print(f"\n  of {len(cast)} cast-Spy games: target_self={len(tself)} (win {wr(tself):.0f}%) | "
          f"target_opp={len(topp)} (win {wr(topp):.0f}%) | no-target-logged={len(tnone)} (win {wr(tnone):.0f}%)")
    # biggest reach drop between consecutive stages
    seq = [("ALL", n)] + [(st, sum(1 for x in g if x[st])) for st in stages]
    drops = [(seq[i][0] + "->" + seq[i+1][0], seq[i][1] - seq[i+1][1]) for i in range(len(seq)-1)]
    drops.sort(key=lambda kv: -kv[1])
    print(f"  biggest reach drops: " + ", ".join(f"{k}={v}" for k, v in drops[:3]))


if __name__ == "__main__":
    main()
