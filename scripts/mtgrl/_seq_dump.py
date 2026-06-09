import json, sys

def dump(path):
    decs=[]
    result=None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line=line.strip()
            if line.startswith("RESULT:"):
                result=line.split("RESULT:",1)[1].strip()
            elif line.startswith("REPLAY_DECISION_JSON:"):
                try: decs.append(json.loads(line.split("REPLAY_DECISION_JSON:",1)[1].strip()))
                except: pass
    print("FILE:", path.split("game_logs")[-1])
    print("RESULT:", result, " ndec:", len(decs))
    # mulligan summary: first few decisions
    interesting=[]
    for i,d in enumerate(decs):
        at=d.get("action_type","")
        st=(d.get("selected_text") or "").replace("\n"," ")
        turn=d.get("turn")
        lib=d.get("library_size")
        hand=d.get("hand")
        gy=d.get("graveyard_size")
        # surface: mulligan, land plays, land grant, dig spells, mana dorks, spy cast, target select, dread return, lotleth
        keys=["Balustrade Spy","Land Grant","Lead the Stampede","Winding Way","Dread Return","Lotleth Giant",
              "Lotus Petal","Tinder Wall","Saruli","Quirion","mulligan","Mulligan","keep","Keep","London","bottom"]
        show = at in ("MULLIGAN","SELECT_TARGETS","LONDON_BOTTOM") or any(k in st for k in keys)
        # also show plays of lands
        if at=="PRIORITY" and ("Play " in st or "Cast " in st):
            show=True
        if show:
            interesting.append((i,turn,at,st,lib,gy,hand))
    for (i,turn,at,st,lib,gy,hand) in interesting:
        hs = ",".join(hand) if isinstance(hand,list) else hand
        print(f"  [{i:3d}] t{turn} {at:14s} lib={lib} gy={gy} | {st[:80]}")
    print("---")

for p in sys.argv[1:]:
    dump(p)
