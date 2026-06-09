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
for path in sys.argv[1:]:
    decs,result=load(path)
    # blue sources used right before spy: find Saruli add any / Lotus Petal sac near first spy
    spy_i=None
    for i,d in enumerate(decs):
        if "Cast Balustrade Spy" in (d.get("selected_text") or ""): spy_i=i; break
    print("FILE",path.split('game_logs')[-1][:45],"RES",result,"ndec",len(decs),"firstSpy@",spy_i,"t",decs[spy_i].get("turn") if spy_i is not None else None)
    # show the tail (last 6 decisions) to see how the game closed
    for i in range(max(0,len(decs)-6),len(decs)):
        d=decs[i]
        print(f"   tail[{i}] t{d.get('turn')} {d.get('action_type')} lib={d.get('library_size')} gy={d.get('graveyard_size')} val={d.get('value_score')} | {(d.get('selected_text') or '')[:55]}")
