import json, sys
def load(path):
    decs=[]; result=None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line=line.strip()
            if line.startswith("RESULT:"): result=line.split("RESULT:",1)[1].strip()
            elif line.startswith("REPLAY_DECISION_JSON:"):
                try: decs.append(json.loads(line.split("REPLAY_DECISION_JSON:",1)[1].strip()))
                except: pass
    return decs,result
path=sys.argv[1]
decs,result=load(path)
print("FILE",path.split('game_logs')[-1][:45],"RES",result)
for i,d in enumerate(decs):
    v=d.get("value_score")
    print(f"[{i:3d}] t{d.get('turn')} {d.get('action_type')[:18]:18s} lib={str(d.get('library_size')):3s} val={v} | {(d.get('selected_text') or '')[:45]}")
