import json, sys
LANDS={"Forest","Swamp","Island","Mountain","Plains"}
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
spy_i=None
for i,d in enumerate(decs):
    if "Cast Balustrade Spy" in (d.get("selected_text") or ""):
        spy_i=i; break
print("FILE",path.split("game_logs")[-1][:60],"RESULT",result)
lo=max(0,spy_i-1); hi=min(len(decs),spy_i+8)
for i in range(lo,hi):
    d=decs[i]
    lib=d.get("library")
    landsin=sum(1 for c in lib if c in LANDS) if isinstance(lib,list) else "?"
    libtail=[c for c in lib if c in LANDS][:5] if isinstance(lib,list) else []
    print(f"[{i}] t{d.get('turn')} {d.get('action_type')} lib={d.get('library_size')} lands={landsin}{libtail} | {(d.get('selected_text') or '')[:60]}")
