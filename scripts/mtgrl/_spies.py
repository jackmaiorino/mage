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
for path in sys.argv[1:]:
    decs,result=load(path)
    print("FILE",path.split("game_logs")[-1][:50],"RESULT",result)
    ncast=0; nlandgrant=0; nquirion=0; ndread=0; nlotleth=0
    for i,d in enumerate(decs):
        st=d.get("selected_text") or ""
        if "Cast Balustrade Spy" in st:
            ncast+=1
            lib=d.get("library"); ll=sum(1 for c in lib if c in LANDS) if isinstance(lib,list) else "?"
            print(f"   SPY#{ncast} @dec{i} t{d.get('turn')} libBefore={d.get('library_size')} landsBefore={ll}")
        if "Land Grant" in st and "Cast" in st: nlandgrant+=1
        if "Quirion" in st: nquirion+=1
        if "Dread Return" in st or "Flashback" in st: ndread+=1
        if st.strip()=="Lotleth Giant": nlotleth+=1
    print(f"   totals: spyCasts={ncast} landGrantCasts={nlandgrant} quirionDecs={nquirion} dread/flashback={ndread} lotlethTargetPicks={nlotleth}")
