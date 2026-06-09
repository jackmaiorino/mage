#!/usr/bin/env python3
"""Spy-combo ORCHESTRATION metric (the real progress signal, not just winrate).

The dominant misplay (verified 2026-06-09): the model self-mills with Balustrade Spy
WITHOUT a 3-creature board to flashback Dread Return, then decks out. So of cast-Spy
games, the key number is what fraction can EXECUTE the finisher (no_board = the misplay).
Combo wins ~88% when the finisher fires; ~82% loss when no board. Track no_board falling
as training learns develop-board-THEN-mill.

Signatures verified from logs: cast_spy = selected_text=='Cast Balustrade Spy';
finisher = selected_text/candidate in {'Flashback sacrifice three','Cast Dread Return',
'Dread Return'}. Reads a cp7 eval run dir (matchups.csv -> full-n winrate; game_logs ->
orchestration over the ~50 logged games). NOTE: the funnel's '(you)' target tag and
'Lotleth Giant' late signatures are UNRELIABLE -- do not use them.
"""
import json
import glob
import csv
import argparse
from pathlib import Path

FIN = ("Flashback sacrifice three", "Cast Dread Return", "Dread Return")


def parse(path):
    decs = []
    outcome = "?"
    for line in open(path, encoding="utf-8", errors="replace"):
        if "REPLAY_DECISION_JSON:" in line:
            try:
                decs.append(json.loads(line.split("REPLAY_DECISION_JSON:", 1)[1].strip()))
            except (ValueError, IndexError):
                pass
        elif line.startswith("RESULT:"):
            t = line.split(":", 1)[1].strip().upper()
            outcome = "win" if t.startswith("WIN") else ("loss" if t.startswith("LOS") else t)
    return outcome, decs


def metric(run_dir, label=""):
    run_dir = Path(run_dir)
    fw = ft = 0
    mc = run_dir / "matchups.csv"
    if mc.exists():
        for r in csv.DictReader(open(mc)):
            try:
                fw += int(r.get("wins") or 0)
                ft += int(r.get("total") or 0)
            except (TypeError, ValueError):
                pass
    logs = glob.glob(str(run_dir / "game_logs" / "**" / "*.txt"), recursive=True)
    ngames = nwin = ncast = navail = ndone = nnoboard = ncast_win = nnoboard_loss = 0
    for f in logs:
        oc, decs = parse(f)
        ngames += 1
        if oc == "win":
            nwin += 1
        spy = [i for i, d in enumerate(decs) if d.get("selected_text", "") == "Cast Balustrade Spy"]
        if not spy:
            continue
        ncast += 1
        if oc == "win":
            ncast_win += 1
        post = decs[spy[0]:]
        done = any(any(w in (d.get("selected_text", "") or "") for w in FIN) for d in post)
        avail = done or any(any(w in str(c) for w in FIN)
                            for d in post for c in (d.get("candidate_texts") or []))
        if done:
            ndone += 1
        if avail:
            navail += 1
        else:
            nnoboard += 1
            if oc == "loss":
                nnoboard_loss += 1
    wr = (fw / ft) if ft else ((nwin / ngames) if ngames else 0.0)
    cast = (ncast / ngames) if ngames else 0.0
    avail_r = (navail / ncast) if ncast else 0.0
    noboard_r = (nnoboard / ncast) if ncast else 0.0
    done_r = (ndone / ncast) if ncast else 0.0
    print(f"{label:10s} wr={wr:.3f}(n={ft or ngames}) cast_spy={cast:.0%} "
          f"finisher_avail={avail_r:.0%} NO_BOARD={noboard_r:.0%} executed={done_r:.0%} "
          f"(logged_cast={ncast})")
    return dict(wr=wr, cast=cast, finisher_avail=avail_r, no_board=noboard_r, executed=done_r,
                logged_cast=ncast)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    metric(a.run_dir, a.label)
