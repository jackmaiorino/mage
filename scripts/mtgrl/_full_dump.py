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
print("RESULT",result,"ndec",len(decs))
for i,d in enumerate(decs):
    at=d.get("action_type","")
    st=(d.get("selected_text") or "").replace("\n"," ")
    turn=d.get("turn"); lib=d.get("library_size"); gy=d.get("graveyard_size")
    hand=d.get("hand"); cand=d.get("candidate_texts")
    nc=len(cand) if isinstance(cand,list) else 0
    hs=",".join(hand) if isinstance(hand,list) else ""
    print(f"[{i:3d}] t{turn} {at} lib={lib} gy={gy} ncand={nc}")
    print(f"      SEL: {st[:90]}")
    if at in ("MULLIGAN",) or i< int(sys.argv[2] if len(sys.argv)>2 else 0):
        print(f"      HAND: {hs}")
