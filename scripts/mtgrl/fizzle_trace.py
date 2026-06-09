import json, os, glob, sys

LANDS = {"Forest", "Swamp"}
BASE = "C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
SUB = "game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"

def find(fname):
    for dd in os.listdir(BASE):
        gdir = os.path.join(BASE, dd, SUB)
        if not os.path.isdir(gdir): continue
        for f in os.listdir(gdir):
            if f.startswith(fname) or f == fname:
                return os.path.join(gdir, f)
    return None

def parse(path):
    decs, result = [], None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("REPLAY_DECISION_JSON:"):
                try: decs.append(json.loads(s[len("REPLAY_DECISION_JSON:"):].strip()))
                except Exception: pass
            elif s.startswith("RESULT:"): result = s.split("RESULT:")[1].strip()
    return decs, result

def main():
    fname = sys.argv[1]
    path = find(fname)
    decs, result = parse(path)
    print(f"FILE {os.path.basename(path)}  RESULT={result}  decisions={len(decs)}")
    spy_idxs = [i for i,d in enumerate(decs)
                if "Balustrade Spy" in (d.get("selected_text") or "")
                and "cast" in (d.get("selected_text") or "").lower()]
    spy_i = spy_idxs[0]
    # Print all PlayerRL1 decisions from turn 1 up to & including the spy cast,
    # showing hand, available land-relevant candidate actions, and what was chosen.
    print(f"\n--- decisions up to spy cast (idx {spy_i}, turn {decs[spy_i].get('turn')}) ---")
    for i in range(0, spy_i+2):
        d = decs[i]
        at = d.get("action_type")
        sel = d.get("selected_text")
        cands = d.get("candidate_texts") or []
        hand = d.get("hand") or []
        lib = d.get("library") or []
        lands_lib = [c for c in lib if c in LANDS]
        lands_hand = [c for c in hand if c in LANDS]
        # flag land-clearing options in candidates
        land_opts = [c for c in cands if any(k in c for k in
                     ["Play Forest","Play Swamp","Land Grant","Forestcycling","Swampcycling","Quirion Ranger","Generous Ent"])]
        flags = []
        if lands_hand: flags.append(f"LANDS_IN_HAND={lands_hand}")
        if land_opts: flags.append(f"LAND_OPTS={land_opts}")
        marker = "  >>> SPY CAST" if i == spy_i else ""
        print(f"[{i:>3}] t{d.get('turn')} {at:<26} sel='{sel}'{marker}")
        print(f"       hand={hand}")
        print(f"       landsInLib={len(lands_lib)}{lands_lib}  {' | '.join(flags)}")

if __name__ == "__main__":
    main()
