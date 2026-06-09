import json, os, glob, re

LANDS = {"Forest", "Swamp"}
BASE = "C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
DIRS = ["baseline_auc_ab", "baseline_auc_s9999", "baseline_auc_big5151"]
SUB = "game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"
RL_STATE = re.compile(r"PlayerRL1 L(\d+) H\d+(?:\[[^\]]*\])? B\d+(?:\[[^\]]*\])? G(\d+)")

# A clearing option removes a land from library:
#  - Forestcycling / Swampcycling (tutor a basic out of library to hand)
#  - Cast Land Grant (fetch a Forest from library to hand)
#  - Quirion Ranger return-a-Forest (returns Forest from battlefield - does NOT remove from lib; but if a Forest is in lib, Quirion can't help directly)
# We focus on the cycling + Land Grant tutors, which directly pull a land OUT of the library.
def clears_lib_land(cand):
    return ("Forestcycling" in cand or "Swampcycling" in cand or
            "Cast Land Grant" in cand)

def parse(path):
    decs, result, states, spy_lines = [], None, [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh):
            s = line.strip()
            if s.startswith("REPLAY_DECISION_JSON:"):
                try: decs.append((ln, json.loads(s[len("REPLAY_DECISION_JSON:"):].strip())))
                except Exception: pass
            elif s.startswith("RESULT:"): result = s.split("RESULT:")[1].strip()
            else:
                m = RL_STATE.search(line)
                if m: states.append((ln, int(m.group(1)), int(m.group(2))))
                if "SELECTED: Cast Balustrade Spy" in s: spy_lines.append(ln)
    return decs, result, states, spy_lines

def main():
    rows = []
    for dd in DIRS:
        gdir = os.path.join(BASE, dd, SUB)
        if not os.path.isdir(gdir): continue
        for path in sorted(glob.glob(os.path.join(gdir, "*.txt"))):
            decs, result, states, spy_lines = parse(path)
            spy_idxs = [i for i,(ln,d) in enumerate(decs)
                        if "Balustrade Spy" in (d.get("selected_text") or "")
                        and "cast" in (d.get("selected_text") or "").lower()]
            if not spy_idxs: continue
            di = spy_idxs[0]; cast_ln, cd = decs[di]
            lib = cd.get("library", []); lands_lib = sum(1 for c in lib if c in LANDS)
            # target
            target = None
            for j in range(di+1, min(di+4, len(decs))):
                if decs[j][1].get("action_type") == "SELECT_TARGETS":
                    target = decs[j][1].get("selected_text"); break
            self_t = target is not None and ("PlayerRL1" in target or "you" in target.lower())
            # milled out?
            gy_before=None
            for (ln,l,g) in states:
                if ln<cast_ln: gy_before=g
            after=[(ln,l,g) for (ln,l,g) in states if ln>cast_ln]
            max_gy=max((g for (_,_,g) in after),default=None)
            milled_out = max_gy is not None and max_gy>=44
            # only self-target fizzles
            if not self_t or milled_out: continue
            # clearing option available AT the spy-cast decision?
            cands_at_cast = cd.get("candidate_texts") or []
            clear_now = [c for c in cands_at_cast if clears_lib_land(c)]
            # clearing option available at ANY earlier own-priority decision in same turn or game (after lands<=lands_lib)?
            cast_turn = cd.get("turn")
            clear_seen_anytime = False
            land_in_hand_anytime = False
            for k in range(0, di):
                _, dk = decs[k]
                ck = dk.get("candidate_texts") or []
                if any(clears_lib_land(c) for c in ck): clear_seen_anytime = True
                hk = dk.get("hand") or []
                if any(c in LANDS for c in hk) and any("Play Forest" in c or "Play Swamp" in c for c in ck):
                    land_in_hand_anytime = True
            rows.append(dict(file=dd[9:]+"/"+os.path.basename(path)[14:23], turn=cast_turn,
                             lands_lib=lands_lib, clear_now=bool(clear_now),
                             clear_now_opts=clear_now, clear_anytime=clear_seen_anytime,
                             milled_out=milled_out, result=result))
    print(f"SELF-target fizzles analyzed: {len(rows)}")
    avoid_now = [r for r in rows if r["clear_now"]]
    avoid_any = [r for r in rows if r["clear_anytime"]]
    print(f"  had a library-land CLEAR option (cycling/LandGrant) AT the Spy-cast decision: {len(avoid_now)}")
    print(f"  had a clear option at SOME earlier decision: {len(avoid_any)}")
    print(f"{'file':<24} {'turn':>4} {'landsLib':>8} {'clearNow':>8} {'clearAny':>8} {'res':>5}  clearOpts")
    for r in sorted(rows, key=lambda x:(not x['clear_now'], x['lands_lib'])):
        opts = ";".join(["Fcyc" if "Forestcycling" in o else "Scyc" if "Swampcycling" in o else "LandGrant" for o in r['clear_now_opts']])
        print(f"{r['file']:<24} {str(r['turn']):>4} {r['lands_lib']:>8} {str(r['clear_now']):>8} {str(r['clear_anytime']):>8} {r['result']:>5}  {opts}")

if __name__ == "__main__":
    main()
