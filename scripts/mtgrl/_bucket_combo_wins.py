import json, os, glob, sys

BASE = "C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
RUNS = ["baseline_auc_ab","baseline_auc_s9999","baseline_auc_big5151"]
MATCH = "Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"

def parse_game(path):
    decs = []  # our (PlayerRL1) decisions
    result = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line=line.strip()
            if line.startswith("RESULT:"):
                result = line.split("RESULT:",1)[1].strip()
            elif line.startswith("REPLAY_DECISION_JSON:"):
                try:
                    j = json.loads(line.split("REPLAY_DECISION_JSON:",1)[1].strip())
                    decs.append(j)
                except Exception:
                    pass
    return decs, result

def classify(decs):
    cast_spy = False
    min_lib_after_spy = None
    spy_turn = None
    milled_out = False
    # find a Spy cast
    saw_cast_idx = None
    for i,d in enumerate(decs):
        st = (d.get("selected_text") or "")
        if "Balustrade Spy" in st and ("Cast" in st):
            cast_spy = True
            if spy_turn is None:
                spy_turn = d.get("turn")
            saw_cast_idx = i
    # track library_size minimum over whole game (after a cast)
    if cast_spy:
        for d in decs:
            ls = d.get("library_size")
            if ls is not None and ls <= 1:
                milled_out = True
    return cast_spy, milled_out, spy_turn

rows=[]
for run in RUNS:
    dir = os.path.join(BASE, run, "game_logs", MATCH)
    for f in sorted(glob.glob(os.path.join(dir, "*.txt"))):
        decs, result = parse_game(f)
        cast, milled, spyturn = classify(decs)
        rows.append((run, os.path.basename(f), result, cast, milled, spyturn, len(decs)))

won_combo = [r for r in rows if r[2]=="WIN" and r[3] and r[4]]
print("TOTAL games:", len(rows))
from collections import Counter
print("results:", Counter(r[2] for r in rows))
print("cast spy:", sum(1 for r in rows if r[3]))
print("milled out (lib<=1 after cast):", sum(1 for r in rows if r[3] and r[4]))
print("WON+cast+milled:", len(won_combo))
print()
for r in won_combo:
    print(f"{r[0]}/{r[1]} spyturn={r[5]} ndec={r[6]}")
